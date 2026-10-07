//! 拉面杯 Game trait 实现
//!
//! 实现回合推进、动作列表、事件处理、训练计算等核心游戏流程。
//!
//! 阶段流转设计：
//! - `RamenStage::next()`：负责回合内普通阶段流转（Begin → Distribute → Train → AfterTrain）
//! - `Game::next()`：负责跨阶段流转（AfterTrain → NextTurn → Begin/特殊阶段），
//!   以及 turn 2 的 `Begin → RegionSelect → BeginAfterRegionSelect` 分叉

use anyhow::{Result, anyhow};
use colored::Colorize;
#[cfg(feature = "cli")]
use comfy_table::{ColumnConstraint, Table, Width};
use rand::{Rng, SeedableRng, prelude::IndexedRandom, rngs::StdRng};
use rand_distr::Distribution;

use super::{
    FeelingType,
    Operation,
    RamenAction,
    RamenGame,
    RamenStage,
    effects::{RamenTrainingEffect, apply_ramen_training_effect, calc_ramen_training_effect},
    events::assign_train_feeling_type,
    rules::{self, get_turn_special_feeling}
};
use crate::{
    diag,
    game::{
        BasePerson,
        FriendOutState,
        PersonType,
        traits::{Game, Person, Trainer},
        uma::Uma
    },
    gamedata::{ActionValue, EventData, GAMECONFIG, GAMECONSTANTS, RamenRegionStrategy, TriggerType, ramen::RAMENDATA},
    global,
    rng::{CLONE_REGION_TAG, fork_local_stream},
    utils::{AttributeArray, global_event_distribution, global_events, system_event, system_event_prob}
};

impl Game for RamenGame {
    type Person = BasePerson;
    type Action = RamenAction;

    /// 初始化人头：开局仅加入非友人卡支援卡和理事长
    ///
    /// 友人卡、NPC和记者在后续回合动态添加（见 `run_stage` Begin 阶段）
    fn init_persons(&mut self) -> Result<()> {
        // 非友人卡支援卡（card_type < 5）
        let persons = self
            .deck
            .iter()
            .filter(|card| card.card_type < 5)
            .map(|card| BasePerson::try_from(card))
            .collect::<Result<Vec<_>>>()?;
        for p in persons {
            self.add_person(p);
        }
        // 理事长
        self.add_person(BasePerson::yayoi());
        Ok(())
    }

    fn turn(&self) -> i32 {
        self.base.turn
    }

    fn max_turn(&self) -> i32 {
        77
    }

    /// 阶段推进
    ///
    /// 回合内流转由 `RamenStage::next()` 处理（Begin → Distribute → Train → AfterTrain）。
    /// 本方法负责 AfterTrain → NextTurn、NextTurn 的回合边界，以及 turn 2 的
    /// `Begin → RegionSelect → BeginAfterRegionSelect` 分叉。
    fn next(&mut self) -> bool {
        // RamenSelect 阶段：
        // - combined_decision=true（合并决策路径，由 apply_combined_ramen_decision 写入）→ 直接推 Train
        //   并在切换时立即触发 ground_ramen_effects（合并决策已包含 ramen + targets 两个决策）
        // - 否则按 pending_ramen 决定推 SpecialSelect（吃了面）还是 Train（不吃面）
        if self.stage == RamenStage::RamenSelect {
            if self.ramen.combined_decision {
                // 合并决策：先切到 Train，再立即 ground（避免下次 next() 还要检查 combined_decision）
                self.stage = RamenStage::Train;
                if self.ground_ramen_effects_with_strategy() {
                    crate::diag!("合并决策 ground_ramen_effects 失败");
                }
                self.ramen.combined_decision = false;
                return true;
            }
            self.stage = if self.ramen.pending_ramen.is_some() {
                RamenStage::SpecialSelect
            } else {
                RamenStage::Train
            };
            return true;
        }

        // SpecialSelect → Train 转换时：触发吃面效果落地
        // 此时 ramen（是否吃）+ special_targets（隐藏风味用法）都已确定，立即生效：
        // 消耗诀窍 / PT 增量 / 生成分身 / 羁绊效果 / 显示 buff + distribution
        // 这样玩家在选训练动作前能看到完整效果。
        if self.stage == RamenStage::SpecialSelect {
            if self.ground_ramen_effects_with_strategy() {
                crate::diag!("ground_ramen_effects 失败");
            }
            self.stage = RamenStage::Train;
            return true;
        }

        // turn 2：Begin 前半段结束后进入地区选择（阶段边界，可供搜索落根）
        if self.stage == RamenStage::Begin && self.base.turn == 2 {
            self.stage = RamenStage::RegionSelect;
            return true;
        }

        // 回合内普通阶段：委托给 RamenStage::next()
        if let Some(mut next_stage) = self.stage.next() {
            // 短路：回合 0-1（剧本机制未启用）或超级拉面回合(72-77，超级拉面自动生效)
            // 直接跳过 RamenSelect/SpecialSelect，只把训练选择权交给 Trainer
            // （Distribute → Train，省略 RamenSelect 中间步骤）。
            if self.stage == RamenStage::Distribute
                && next_stage == RamenStage::RamenSelect
                && (self.base.turn < 2 || self.is_super_ramen_turn())
            {
                next_stage = RamenStage::Train;
            }
            self.stage = next_stage;
            return true;
        }

        // AfterTrain → NextTurn（RamenStage::next() 返回 None 时）
        if self.stage == RamenStage::AfterTrain {
            self.stage = RamenStage::NextTurn;
            return true;
        }

        // NextTurn：回合边界逻辑
        if self.stage == RamenStage::NextTurn {
            // 吃面 PT 增量 / eat_count += 1 延后到此阶段（在 `clear current_ramen`
            // 之前），保证训练阶段的 `calc_ramen_training_effect` 用吃面前的
            // `scenario_pt` 算 ramen_pt_effect / region_bonus 档位，PT 增量从
            // 下一回合才参与档位计算。
            if let Some(ramen_idx) = self.ramen.current_ramen {
                let year_idx = (self.current_year() - 1) as usize;
                // `next()` 返回 bool，不能 `?`；year_idx 在剧本三年内必合法，
                // 此处仅做防御性 fallback。
                match super::rules::calc_ramen_pt_gain(year_idx, self.ramen.eat_count) {
                    Ok(pt_gain) => {
                        self.ramen.scenario_pt += pt_gain;
                        self.ramen.eat_count += 1;
                        crate::diag!(
                            ">> 吃面[{}] PT+{} (NextTurn 后置, 总计{})",
                            ramen_idx,
                            pt_gain,
                            self.ramen.scenario_pt,
                        );
                    }
                    Err(e) => {
                        crate::diag!(
                            ">> 吃面[{}] PT 增量计算失败 (year_idx={}, eat_count={}): {}",
                            ramen_idx,
                            year_idx,
                            self.ramen.eat_count,
                            e,
                        );
                    }
                }
            }

            // 清除当前回合的吃面状态
            self.ramen.current_ramen = None;
            // 防御性清空 pending
            self.ramen.clear_pending();

            // RMJ 结算回合检查
            if self.is_rmj_turn() {
                let year_idx = (self.current_year() - 1) as usize;
                // 观测归档：必须在 live 计数器清零之前写入当年 PT / 吃面次数。
                // 年份用 RMJ 回合硬编码（23→0, 47→1, 71→2），不用 current_year() 再推一次。
                // 注意：同一 turn 23 地区选择归档的是第 2 年，与此处「结算第 1 年」不是同一下标。
                match super::RamenState::rmj_archive_year_idx(self.base.turn) {
                    Ok(idx) => {
                        if let Err(e) = self.ramen.archive_year_counters(idx) {
                            crate::diag!("逐年 PT/吃面归档失败: {e}");
                        }
                    }
                    Err(e) => {
                        crate::diag!("逐年 PT/吃面归档失败: {e}");
                    }
                }
                let result = rules::check_rmj(&mut self.ramen, year_idx);
                if result.is_success() {
                    self.ramen.train_level_bonus += 1;
                }
                diag!(
                    "RMJ 结算: {:?} (PT={}) 训练等级加成={}",
                    result,
                    self.ramen.scenario_pt,
                    self.ramen.train_level_bonus
                );
                self.ramen.eat_count = 0;
                // RMJ 事件立即 apply（在 turn=N 末触发，而非 turn=N+1 末）
                // 原因：push 到 unresolved_events 后会被 AfterTrain 阶段消费，
                // 而 AfterTrain 阶段在 turn=N 的 NextTurn 阶段之后才轮到 turn=N+1，
                // 会延迟一整个回合。
                // RMJ 事件没有 player_select=true，可以直接 apply 而不需 Trainer。
                // 事件 ID：401404(年1) / 401405(年2) / 401406(年3)，按 rmj_results[year_idx] 决定 result=2/1
                if let Some(event) = find_rmj_event(year_idx) {
                    diag!("+ 事件: #{} {} (回合 {} 末)", event.id, event.name, self.base.turn + 1);
                    // 回合固定流（RMJ 固定触发，v2 §4.3）；未注入 rule_master 时
                    // 回退旧 internal_rng（再未注入则 os rng），保持改造前可复现性
                    let err = match self.turn_fixed.take() {
                        Some(mut f) => {
                            let e = self.apply_event(&event, 0, &mut f).is_err();
                            self.turn_fixed = Some(f);
                            e
                        }
                        None => match self.internal_rng.take() {
                            Some(mut r) => {
                                let e = self.apply_event(&event, 0, &mut r).is_err();
                                self.internal_rng = Some(r);
                                e
                            }
                            None => self.apply_event(&event, 0, &mut StdRng::from_os_rng()).is_err()
                        }
                    };
                    if err {
                        crate::diag!("RMJ 事件 #{} apply 失败: {:?}", event.id, event.name);
                    }
                }
                // RMJ 结算后 scenario_pt 归零，下一年重新累计
                // 此时 rmj_results 已写入，下一年的 ramen_success_effect / ramen_fail_effect 已可读取
                let pt_before_reset = self.ramen.scenario_pt;
                self.ramen.scenario_pt = 0;
                diag!("scenario_pt 已归零（结算前 PT={}，下年重新累计）", pt_before_reset);
            }

            // 年度地区选择：回合23（第1年结束后）、回合47（第2年结束后）
            // RMJ 结算后选择下一年的地区
            match self.base.turn {
                23 | 47 => {
                    self.stage = RamenStage::RegionSelect;
                    return true;
                }
                _ => {}
            }

            // 特殊阶段跳转：超级拉面选择（回合71）
            if self.base.turn == 71 {
                self.stage = RamenStage::SuperRamenSelect;
                return true;
            }

            // 推进到下一回合
            return self.advance_turn();
        }

        // 特殊阶段：turn 2 的 RegionSelect 回到 Begin 后半段；
        // turn 23/47 的 RegionSelect 以及 SuperRamenSelect / Settlement 推进到下一回合。
        // 绝不能让 turn 2 走 advance_turn()，否则会整段跳过 turn 2。
        if self.stage == RamenStage::RegionSelect {
            if self.base.turn == 2 {
                self.stage = RamenStage::BeginAfterRegionSelect;
                return true;
            }
            return self.advance_turn();
        }
        if matches!(
            self.stage,
            RamenStage::SuperRamenSelect | RamenStage::Settlement
        ) {
            return self.advance_turn();
        }

        false
    }

    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        match self.stage {
            RamenStage::Begin => self.run_begin(trainer, rng)?,
            RamenStage::BeginAfterRegionSelect => self.run_begin_suffix(trainer, rng)?,
            RamenStage::Distribute => self.run_distribute(rng)?,
            RamenStage::RamenSelect => self.run_ramen_select(trainer, rng)?,
            RamenStage::SpecialSelect => self.run_special_select(trainer, rng)?,
            RamenStage::Train => self.run_train(trainer, rng)?,
            RamenStage::AfterTrain => self.run_after_train(trainer, rng)?,
            RamenStage::NextTurn => {} // 回合推进逻辑在 next() 中处理
            RamenStage::RegionSelect => {
                // 回合2→第1年(year_idx=0)，回合23→第2年(year_idx=1)，回合47→第3年(year_idx=2)
                // 与逐年归档共用同一份映射，避免两处真值来源漂移
                let year_idx = super::RamenState::region_archive_year_idx(self.base.turn)?;
                self.run_region_select(trainer, rng, year_idx)?;
            }
            RamenStage::SuperRamenSelect => self.run_super_ramen_select(trainer, rng)?,
            RamenStage::Settlement => {} // RMJ 结算在 next() 中处理
        }
        Ok(())
    }

    fn list_actions(&self) -> Result<Vec<Self::Action>> {
        // SuperRamenSelect / RegionSelect 必须在 is_race_turn 短路之前：
        // turn 71 若是比赛回合，否则会错误返回比赛动作；
        // RegionSelect 若落到 `_` 回退会拿到训练+吃面组合，搜索比的是错误动作空间。
        match self.stage {
            RamenStage::SuperRamenSelect => {
                return super::action::list_super_ramen_select_actions();
            }
            RamenStage::RegionSelect => {
                return super::action::list_region_select_actions(self.base.turn);
            }
            RamenStage::Train => return Ok(self.list_train_actions()),
            _ => {}
        }

        // 公共判定：friend_outing / ill（复用 BaseGame 通用规则）
        let can_friend_outing = self.can_friend_outing();
        let is_ill = self.uma.flags.ill;
        let can_race = self.can_self_race();

        // 按当前阶段返回候选动作
        match self.stage {
            RamenStage::RamenSelect => {
                // 拉面回合（turn >= 2 且非超级拉面回合）才有面可选；其他时段只显示"不吃"。
                // 注：`Game::next()` 已在 Distribute 阶段将回合 0-1 / 超级拉面回合直接跳到 Train，
                // 不会进入本分支；此处保留作为防御性回退（应对外部直接 set stage 的场景）。
                if self.base.turn >= 2 && !self.is_super_ramen_turn() {
                    Ok(super::action::list_ramen_select_actions(
                        &self.ramen,
                        &self.ramen.selected_regions
                    ))
                } else {
                    Ok(vec![RamenAction::ramen_select(None)])
                }
            }
            RamenStage::SpecialSelect => {
                let ramen_idx = self
                    .ramen
                    .pending_ramen
                    .ok_or_else(|| anyhow::anyhow!("SpecialSelect 阶段要求 pending_ramen 已设置"))?;
                super::action::list_special_select_actions(&self.ramen, ramen_idx)
            }
            // 其他阶段的 list_actions 保留旧行为（虽然外部不会在此阶段调）
            _ => {
                let available_ramens = if self.base.turn >= 2 && !self.is_super_ramen_turn() {
                    super::action::get_available_ramens(&self.ramen, &self.ramen.selected_regions)
                } else {
                    vec![]
                };
                Ok(super::action::list_all_actions(
                    &available_ramens,
                    can_friend_outing,
                    is_ill,
                    self.is_xiahesu(),
                    can_race
                ))
            }
        }
    }

    fn generate_events(&self, rng: &mut impl Rng) -> Vec<EventData> {
        let mut events = vec![];
        let no_event_turns = &global!(GAMECONSTANTS).no_event_turns;

        // 剧本事件
        let ramen_data = global!(RAMENDATA);
        let story_events: Vec<EventData> = ramen_data
            .scenario_events
            .iter()
            .filter_map(|e| match &e.trigger {
                TriggerType::Random { .. } => Some(e.clone()),
                TriggerType::Code => None,
                TriggerType::Fixed { turns } => {
                    if turns.contains(&self.base.turn) {
                        Some(e.clone())
                    } else {
                        None
                    }
                }
            })
            .collect();
        if !story_events.is_empty() {
            return story_events;
        }

        // 全局剧本事件（400000400 马娘登场 / 4009 经典年新年 / 4010 古马年新年 等）
        // 这些事件是 gamesystem 共享的（onsen/basic 也用），拉面杯需要按 Fixed 回合触发
        let global_story_events: Vec<EventData> = global_events()
            .story_events
            .iter()
            .filter_map(|e| match &e.trigger {
                TriggerType::Random { .. } => Some(e.clone()),
                TriggerType::Code => None,
                TriggerType::Fixed { turns } => {
                    if turns.contains(&self.base.turn) {
                        Some(e.clone())
                    } else {
                        None
                    }
                }
            })
            .collect();
        if !global_story_events.is_empty() {
            return global_story_events;
        }

        if !no_event_turns.contains(&self.base.turn) {
            // 友人出门事件判定已移至 `run_begin`（策略相关随机 → 策略流，v2 §4.3）：
            // 若留在此处消费回合固定流，固定流的消耗量会随策略（是否点击友人）
            // 变化，导致角标/分布/hint 跨策略错位。
            // 一般随机事件
            match global_event_distribution().sample(rng) {
                0 => {
                    // 只从 Card 类型的人物中随机选择
                    let card_indices: Vec<i32> = self
                        .persons
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.person_type == PersonType::Card)
                        .map(|(i, _)| i as i32)
                        .collect();
                    if let Some(&person_index) = card_indices.choose(rng) {
                        if let Some(event) = self.base.generate_card_event(person_index, rng) {
                            events.push(event);
                        }
                    }
                }
                1 => {
                    if let Some(event) = self.base.random_select_event(&global_events().uma_events, rng) {
                        events.push(event);
                    }
                }
                2 => {
                    if self.base.turn >= 12 {
                        events.push(system_event("drop_motivation").expect("掉心情事件").clone());
                    }
                }
                _ => {}
            }
        }
        events
    }

    fn apply_event(&mut self, event: &EventData, choice: usize, rng: &mut impl Rng) -> Result<()> {
        // RMJ 事件特殊处理：根据 rmj_results[year_idx] 选择 result=2 或 result=1 的分支
        if let Some(year_idx) = rmj_event_year(event.id) {
            if let Some(choice_group) = event.choices.first() {
                if let Some(target) =
                    select_rmj_choice_by_result(choice_group, self.ramen.rmj_results.get(year_idx).copied())
                {
                    diag!("RMJ 事件 #{} 应用 result={} 分支", event.id, target.result);
                    self.base.uma.add_value(&target.value);
                } else {
                    diag!(
                        "RMJ 事件 #{} 无法匹配 result 分支（rmj_results[{}]={:?}），使用默认分支",
                        event.id,
                        year_idx,
                        self.ramen.rmj_results.get(year_idx)
                    );
                }
            }
            // 计数 +1（与 base.apply_event 行为一致）
            self.base.events.entry(event.id).and_modify(|x| *x += 1).or_insert(1);
            return Ok(());
        }

        if let Some(result) = self.base.apply_event(event, choice, rng) {
            if let Some(person_index) = &event.person_index
                && result.value.friendship != 0
            {
                self.add_friendship(*person_index as usize, result.value.friendship);
            }
        }
        match event.id {
            4012 | 4013 => {
                let inherit_value = ActionValue {
                    status_pt: self.inherit.inherit(rng),
                    ..Default::default()
                };
                let inherit_limit = self.inherit.inherit_limit(rng);
                self.uma.add_value(&inherit_value);
                self.uma.five_status_limit.add_eq(&inherit_limit);
            }
            5007 => {
                if rng.random_bool(system_event_prob("qiezhe_normal")?) {
                    diag!(">> 获得【切者】");
                    self.uma.flags.gain_qiezhe();
                }
            }
            super::events::EVENT_FRIEND_UNLOCK => {
                diag!(">> 友人出行已解锁");
                self.friend.out_state = FriendOutState::AfterUnlock;
                self.uma.flags.refresh_mind = 1;
            }
            _ => {}
        }
        Ok(())
    }

    // ========== Getters ==========

    fn persons(&self) -> &[Self::Person] {
        &self.persons
    }
    fn persons_mut(&mut self) -> &mut [Self::Person] {
        &mut self.persons
    }
    fn absent_rate_drop(&self) -> i32 {
        self.base.absent_rate_drop
    }
    fn distribution(&self) -> &Vec<Vec<i32>> {
        &self.base.distribution
    }
    fn distribution_mut(&mut self) -> &mut Vec<Vec<i32>> {
        &mut self.base.distribution
    }
    fn uma(&self) -> &Uma {
        &self.uma
    }
    fn uma_mut(&mut self) -> &mut Uma {
        &mut self.uma
    }
    fn deck(&self) -> &Vec<crate::game::SupportCard> {
        &self.deck
    }

    fn deyilv(&mut self, person_index: i32) -> f32 {
        // 人头下标 ≠ 卡组下标：负数、越界、无卡人头（理事长 / 记者 / NPC）一律返回 0
        let Ok(person_index) = usize::try_from(person_index) else {
            return 0.0;
        };
        let Some(di) = Game::deck_index_of(self, person_index) else {
            return 0.0;
        };
        // `calc_training_effect` 返回 owned cumulative effect——先取出 `deyilv` 后直接
        // `move` 进 `self.deck[i].effect`（避免 `.clone()`）。features.rs NN 输入读
        // `card.effect` 取累计 deyilv 与此一致。
        let eff = self.deck[di].calc_training_effect(self, 0);
        let deyilv = eff.deyilv;
        self.deck[di].effect = eff;
        // is_locked 字段保留（NN feature 兼容），每次 deyilv 调用都标记
        self.deck[di].is_locked = true;
        // 卡得意率 + 剧本得意率总加成（参见 calc_scenario_deyilv）
        let scenario_deyilv = super::effects::calc_scenario_deyilv(self);
        deyilv + scenario_deyilv as f32
    }

    fn has_group_buff(&self) -> bool {
        self.friend.group_buff_turn > 0
    }

    /// 重写闪耀判定
    ///
    /// 支援卡（含分身）：只能在本体的得意训练位置闪耀（train_type == train && friendship >= 80）
    /// 友人卡：有 group buff 时闪耀
    fn is_shining_at(&self, person_index: usize, train: usize) -> bool {
        if person_index >= self.persons.len() {
            return false;
        }
        let person = &self.persons[person_index];
        match person.person_type {
            // 支援卡（含分身）：只能在本体的得意训练位置闪耀
            PersonType::Card => person.train_type == train as i32 && person.friendship >= 80,
            // 友人卡：有 group buff 时闪耀
            PersonType::ScenarioCard => self.has_group_buff(),
            // NPC、理事长、记者不能闪耀
            _ => false
        }
    }

    fn train_level(&self, train: usize) -> usize {
        if self.is_xiahesu() {
            5
        } else {
            let base = self.base.train_level_count[train] as usize / 4 + 1;
            (base + self.ramen.train_level_bonus as usize).min(5).max(1)
        }
    }

    fn training_basic_value(&self) -> &crate::gamedata::TrainingBasicTable {
        &global!(RAMENDATA).training_basic_value
    }

    fn explain_distribution(&self) -> Result<String> {
        let base_headers = vec!["速", "耐", "力", "根", "智"];
        // 剧本机制已开启 且 非URA回合 时显示诀窍角标
        let show_ramen = self.base.turn >= 2 && !self.is_super_ramen_turn();
        let headers: Vec<String> = base_headers
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                if show_ramen {
                    if let Some(types) = self.ramen.train_feeling_type {
                        format!("{}{:?}", h, types[i])
                    } else {
                        h.to_string()
                    }
                } else {
                    h.to_string()
                }
            })
            .collect();
        // 防御：distribution 未初始化（dist.len() < 5）时填充空 vec，
        // 避免 ground_ramen_effects 在 distribute 之前触发时 panic
        let dist: Vec<Vec<i32>> = if self.base.distribution.len() < 5 {
            let mut d = self.base.distribution.clone();
            while d.len() < 5 {
                d.push(vec![]);
            }
            d
        } else {
            self.base.distribution.clone()
        };
        let mut rows = vec![];
        for i in 0..6 {
            let mut row = vec![];
            for train in 0..5 {
                if let Some(id) = dist[train].get(i) {
                    if *id < 0 || *id as usize >= self.persons.len() {
                        row.push("".to_string());
                        continue;
                    }
                    let p = &self.persons[*id as usize];
                    let shining = self.is_shining_at(*id as usize, train);
                    let text = if colored::control::SHOULD_COLORIZE.should_colorize() {
                        Self::format_person_colored(p, shining)
                    } else {
                        // 无色环境（no-color / 非 tty）保留原标记：彩圈 +X+、hint !
                        let mut t = p.explain();
                        if shining {
                            t = format!("+{t}+");
                        }
                        t
                    };
                    row.push(text);
                } else {
                    row.push("".to_string());
                }
            }
            rows.push(row);
        }
        // cli 下输出完整表格 + 训练数值计算明细；core-only 下退化为简化文本
        #[cfg(feature = "cli")]
        {
            let mut table = Table::new();
            table.set_header(headers.clone()).add_rows(rows).set_width(80);
            for col in table.column_iter_mut() {
                col.set_constraint(ColumnConstraint::Absolute(Width::Percentage(20)));
            }
            let mut lines = vec![table.to_string()];
            // 训练数值计算明细（每训练位一行：数值 + 失败率 + 诀窍槽 A/B/C 增量）
            // ——玩家手动玩时此为决策依据；调用 `collect_train_lines` 输出 5 行
            //   紧跟在人头分布表之后，序列化为文字段落（cli 兼容）
            self.collect_train_lines(&mut lines, &headers, &dist, show_ramen)?;
            Ok(lines.join("\n"))
        }
        #[cfg(not(feature = "cli"))]
        {
            let mut lines = vec![];
            for (i, row) in rows.iter().enumerate() {
                lines.push(format!("[{}] {}", i, row.join(" ")));
            }
            // core-only 也保留训练数值计算明细（褪化为文本，仍可用于结构化日志）
            self.collect_train_lines(&mut lines, &headers, &dist, show_ramen)?;
            Ok(lines.join("\n"))
        }
    }

    fn calc_training_value(&self, buffs: &crate::game::CardTrainingEffect, train: usize) -> Result<ActionValue> {
        if train > 5 {
            return Err(anyhow!("训练类型错误"));
        }
        // 完整实现（与共享上层公式逐位等价，见守门测试
        // `test_train_eval_deterministic_and_cached_consistent` 守门 3）：
        // trait 方法保持单函数体——拆成"薄壳→inherent"会让 microbench C/D 段
        // （走 trait 路径）因跨 impl 边界未内联带回 +15ns/train 的测量口径退化；
        // eval_train 保存下层属性，并通过 apply_ramen_training_effect 应用每碗面的效果。
        // 两阶段计算：参考 OnsenGame 的实现
        // 1. 下层值：default_calc_training_value 应用卡 buff（友情/训练/干劲/人数/成长率），
        //    然后约束 status_pt 各元素 ≤ 100（剧本规则：下层不超过 100）
        let mut base_value = self.default_calc_training_value(buffs, train)?;
        for i in 0..6 {
            base_value.status_pt[i] = base_value.status_pt[i].min(100);
        }
        // 2. 拉面 buff：累乘到下层值上（不合并到 buffs，避免累乘 vs 加法混淆）
        let is_shining = self.shining_count(train) > 0;
        let ramen_effect = super::effects::calc_ramen_training_effect(self, train, is_shining);
        let xunlian_mult = (100 + ramen_effect.xunlian) as f64 / 100.0;
        let youqing_mult = (100 + ramen_effect.youqing) as f64 / 100.0;
        let pt_bonus_mult = (100 + ramen_effect.pt_bonus) as f64 / 100.0;
        // 上限口径独立：属性吃 status_limit，PT 只吃 pt_limit（「SP獲得上限+100」）
        let status_limit = 100 + ramen_effect.status_limit;
        let pt_limit = 100 + ramen_effect.pt_limit;
        // 3. 上层值：拉面 buff 带来的增量
        // - xunlian × youqing 对 status_pt[0..4]（5 个属性训练值，含副属性加成 buff.bonus）都生效
        // - pt_bonus 仅对 status_pt[5]（PT）单独生效
        for i in 0..5 {
            if base_value.status_pt[i] > 0 {
                let upper_raw =
                    (base_value.status_pt[i] as f64 * xunlian_mult * youqing_mult) as i32 - base_value.status_pt[i];
                let upper = upper_raw.min(status_limit).max(0);
                base_value.status_pt[i] += upper;
            }
        }
        // PT 部分额外乘 pt_bonus（友情口径与属性一致）
        let pt_upper_raw = (base_value.status_pt[5] as f64 * xunlian_mult * youqing_mult * pt_bonus_mult) as i32
            - base_value.status_pt[5];
        let pt_upper = pt_upper_raw.min(pt_limit).max(0);
        base_value.status_pt[5] += pt_upper;
        Ok(base_value)
    }

    fn person_is_available(&self, person_index: usize) -> bool {
        match self.persons[person_index].person_type {
            PersonType::ScenarioCard => self.base.turn >= 2,
            PersonType::Reporter => self.base.turn >= 12,
            _ => true
        }
    }

    /// 记录本回合判定为「不在」的卡人头下标（供后续剧本机制计算）
    fn record_absent_person(&mut self, person_index: i32) {
        self.ramen.absent_cards.push(person_index);
    }

    fn distribute_hint(&mut self, rng: &mut impl Rng) -> Result<()> {
        let base_hint_rate = global!(GAMECONSTANTS).base_hint_rate / 100.0;
        let hint_bonus_pct = self.calc_hint_bonus_pct() as f64;
        // hint_special 生效时，位于 at_trains 训练位置的所有支援卡 (PersonType::Card) is_hint 都强制为 true
        // 生效条件：当前回合吃了面 + ramen_basic_effect[year].hint_special == true + 支援卡种类>=4
        let hint_special_active = self.calc_hint_special_active(self.ramen.current_ramen);
        let special_trains = if hint_special_active {
            self.calc_hint_special_at_trains(self.ramen.current_ramen)
        } else {
            Default::default()
        };
        // 人头下标 ≠ 卡组下标，按 card_id 查找 Hint 概率加成。
        let deck = &self.base.deck;
        for person in &mut self.persons {
            if person.person_type() == PersonType::Card {
                let bonus = person
                    .card_id()
                    .and_then(|cid| deck.iter().find(|card| card.card_id == cid))
                    .map_or(0, |card| card.card_value().hint_prob_increase);
                let card_bonus = (100 + bonus) as f64 / 100.0;
                let hint_prob = base_hint_rate * card_bonus * (1.0 + hint_bonus_pct / 100.0);
                person.set_hint(rng.random_bool(hint_prob));
            }
        }
        // hint_special：强制设置 at_trains 训练位置所有支援卡的 is_hint
        if hint_special_active && !special_trains.is_empty() {
            for (train_idx, has_person) in self.base.distribution.iter().enumerate() {
                if !special_trains.contains(&(train_idx as i32)) {
                    continue;
                }
                for &person_index in has_person {
                    if person_index < 0 {
                        continue;
                    }
                    if let Some(p) = self.persons.get_mut(person_index as usize) {
                        if p.person_type == PersonType::Card {
                            p.set_hint(true);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl RamenGame {
    /// 列出当前基础局面的训练阶段动作；必赛回合只有比赛，候选面与当前阶段不影响列表。
    pub(crate) fn list_train_actions(&self) -> Vec<RamenAction> {
        if self.is_race_turn() {
            return vec![RamenAction::no_ramen(Operation::Race)];
        }
        super::action::list_train_actions(
            self.can_friend_outing(), self.uma.flags.ill, self.is_xiahesu(), self.can_self_race()
        )
    }

    /// 使用已计算的拉面效果求训练值，与策略评估共用上层加成公式。
    #[inline(always)]
    pub fn calc_training_value_with_effect(
        &self, buffs: &crate::game::CardTrainingEffect, train: usize, ramen_effect: &RamenTrainingEffect
    ) -> Result<ActionValue> {
        if train > 5 {
            return Err(anyhow!("训练类型错误"));
        }
        let mut base_value = self.default_calc_training_value(buffs, train)?;
        base_value.status_pt = apply_ramen_training_effect(base_value.status_pt, ramen_effect);
        Ok(base_value)
    }

    /// 友人解锁事件判定（策略相关随机 → 策略流，v2 §4.3）
    //
    // 从 `generate_events` 移出：触发条件依赖 `friend.out_state`（是否点击友人，
    // 策略相关）。若留在此处消耗回合固定流，固定流的消耗量会随策略变化，
    // 导致角标/分布/hint 跨策略错位。
    fn try_friend_unlock(&self, rng: &mut impl Rng) -> Option<EventData> {
        if global!(GAMECONSTANTS).no_event_turns.contains(&self.base.turn)
            || self.friend.out_state != FriendOutState::BeforeUnlock
        {
            return None;
        }
        let friendship = self.persons[self.friend.person_index as usize].friendship;
        let out_prob = if friendship < 60 {
            system_event_prob("friend_unlock_low")
        } else {
            system_event_prob("friend_unlock_high")
        }
        .expect("friend_unlock_* prob key not found");
        if rng.random_bool(out_prob) {
            let ramen_data = global!(RAMENDATA);
            Some(ramen_data.friend_events["out"].clone())
        } else {
            None
        }
    }

    /// 分布表单元格的彩色呈现（仅在允许颜色时调用）
    ///
    /// 颜色规则：
    /// - 彩圈人物：名字亮绿色，去掉 `+` 标记
    /// - hint 人物：感叹号亮黄色，名字颜色不变
    /// - 友人卡（ScenarioCard）：名字绿色
    ///
    /// 优先级：彩圈（亮绿）> 友人（绿）；hint 感叹号独立叠加。
    fn format_person_colored(p: &BasePerson, shining: bool) -> String {
        let raw = p.explain();
        // 拆分 hint 感叹号与名字（explain() 中 `!` 为前缀）
        let (mark, name) = if p.is_hint {
            ("!".bright_yellow().to_string(), raw.trim_start_matches('!').to_string())
        } else {
            (String::new(), raw)
        };
        let name = if shining {
            name.bright_green().to_string()
        } else if p.person_type == PersonType::ScenarioCard {
            name.green().to_string()
        } else {
            name
        };
        format!("{mark}{name}")
    }

    /// 落地所有"吃面后立即生效"的效果（不含 PT 增量）
    ///
    /// 这是从原 `RamenAction::apply_ramen` + `apply_ramen_friendship` 抽出的统一入口，
    /// 把"选面 + 选隐藏"两个 Trainer 决策之后**所有立即生效**的效果整合到一起。
    ///
    /// 调用时机：
    /// - **三阶段路径**：`SpecialSelect → Train` 过渡时（`Game::next()` 自动触发）
    /// - **合并决策路径**：`RamenSelect → Train` 过渡时（`combined_decision=true`，
    ///   `Game::next()` 自动触发）
    /// - **外部接口**：通信模块传入"已吃面但未训练"的中间状态时手动调用
    ///
    /// 立即生效的效果：
    /// 1. **消耗诀窍**（`consume_for_ramen`）
    /// 2. **设置 `current_ramen`**（标记吃了面，让 `ramen_basic_effect` / `ramen_region_effect` 在训练阶段生效）
    /// 3. **生成分身**（地区拉面 id >= 5 + `deck_can_split`）
    /// 4. **羁绊效果**（吃面或超级拉面回合的 `ramen_basic_effect.friendship`）
    /// 5. **打印 buff 摘要 + distribution**（让玩家在选训练前看到效果）
    ///
    /// **PT 增量和 `eat_count += 1` 延后到 `NextTurn` 阶段**（在 `clear current_ramen`
    /// 之前统一处理）。这样本回合训练阶段的 `calc_ramen_training_effect` 读到的
    /// `scenario_pt` 仍是"吃面前"的 PT，确保 `ramen_pt_effect` / `region_bonus` 档位
    /// 不会因为本次吃面立即跨档抬升（PT 增量从下一回合才参与档位计算）。
    ///
    /// **不执行 `operation`**（训练/比赛/休息等），这是 Train 阶段的职责。
    /// 不执行事件（hint 等），事件在 Train 阶段的 `do_train` 中触发。
    ///
    /// # 参数
    /// - `rng`：随机数生成器（分身分配使用）
    pub fn ground_ramen_effects(&mut self, rng: &mut impl Rng) -> Result<()> {
        // 1. 消耗诀窍 + 设置 current_ramen + 分身（仅当 pending_ramen.is_some()）
        if let Some(ramen_idx) = self.ramen.pending_ramen {
            let targets = self.ramen.pending_special_targets;
            let used_special = super::rules::consume_for_ramen(&mut self.ramen, ramen_idx, &targets)?;
            self.ramen.current_ramen = Some(ramen_idx);

            crate::diag!(
                ">> 吃面[{}]（PT 增量 / eat_count 延后到 NextTurn），消耗隐藏风味{}",
                ramen_idx,
                used_special
            );

            // 生成分身（id >= 5 + deck_can_split）
            Self::distribute_region_clones(self, ramen_idx, rng)?;
        }

        // 2. 羁绊效果（吃面或超级拉面回合）
        Self::apply_ramen_friendship(self)?;

        // 3. 显示 buff + distribution（玩家在选训练前看到效果）
        // 整段 cfg(feature = "diag")：MCTS rollout 编译时关掉 diag feature 可省
        // comfy-table 构造 + ANSI 解析（最贵的展示开销）；? 在 cfg 包块内整段消失。
        // 内层再包 enabled()：diag feature 开启的 umasim 在 MCTS rollout 期间
        // 由 DiagGuard 全局静默，跳过 explain 系列的 String/comfy-table 构造。
        #[cfg(feature = "diag")]
        if crate::output::diagnostic::enabled() {
            crate::diag!("---- 吃面后 ----");
            // 吃面后插入一行马娘状态（诀窍/PT 消耗后的最新状态）
            crate::diag!("{}", self.uma.explain()?);
            let ramen_info = self.explain_ramen_info();
            if !ramen_info.is_empty() {
                crate::diag!("{}", ramen_info);
            }
            if let Ok(dist_info) = self.explain_distribution() {
                crate::diag!("训练:\n{}", dist_info);
            }
        }

        Ok(())
    }

    /// 落地吃面效果（使用策略流）
    ///
    /// 分身分配属策略交互随机（v2 §4.3），RNG 取自/放回 [`Self::strategy`]；
    /// 未注入 rule_master 时回退旧 `internal_rng`（再未注入则 os rng），
    /// 保持与规则层改造前一致的可复现性契约。返回是否出错。
    fn ground_ramen_effects_with_strategy(&mut self) -> bool {
        match self.strategy.take() {
            Some(mut s) => {
                let err = self.ground_ramen_effects(&mut s).is_err();
                self.strategy = Some(s);
                err
            }
            None => match self.internal_rng.take() {
                Some(mut r) => {
                    let err = self.ground_ramen_effects(&mut r).is_err();
                    self.internal_rng = Some(r);
                    err
                }
                None => self.ground_ramen_effects(&mut StdRng::from_os_rng()).is_err()
            }
        }
    }

    /// 拉面羁绊效果（吃面或超级拉面回合触发）
    ///
    /// 从原 `RamenAction::apply_ramen_friendship` 抽出，统一在 `ground_ramen_effects` 中调用。
    /// 生效条件：`current_ramen.is_some()` 或超级拉面回合（72-77）。
    fn apply_ramen_friendship(&mut self) -> Result<()> {
        let eating = self.ramen.current_ramen.is_some();
        let super_ramen = self.is_super_ramen_turn();
        if !eating && !super_ramen {
            return Ok(());
        }
        let year_idx = (self.current_year() - 1) as usize;
        let ramen_data = global!(RAMENDATA);
        if let Some(basic) = ramen_data.ramen_basic_effect.get(year_idx) {
            if basic.friendship > 0 {
                for i in 0..self.persons.len() {
                    if matches!(self.persons[i].person_type, PersonType::Card | PersonType::ScenarioCard) {
                        self.add_friendship(i, basic.friendship);
                    }
                }
            }
        }
        Ok(())
    }

    /// 分配地区拉面分身（id >= 5 时触发）
    ///
    /// 从原 `RamenAction::distribute_clones` 抽出并重命名（避免与
    /// `distribute_super_ramen_clones` 混淆），统一在 `ground_ramen_effects` 中调用。
    ///
    /// 分身分配逻辑：
    /// - 满员规则：每个训练位置最多 5 人；已满则优先挤掉 NPC
    /// - 同一训练不能存在相同卡的 `Person` 和分身
    /// - 分身不计算得意率，不包含友人卡
    /// - **缺席优先**：本回合被判定「不在」的支援卡（`PersonType::Card`）优先按
    ///   缺席顺序补进 `at_trains` 分身位（先缺谁补谁）；全部支援卡都在训练后，
    ///   剩余分身位才随机复制在场支援卡
    fn distribute_region_clones(&mut self, region_id: usize, rng: &mut impl Rng) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        let region = &ramen_data.ramen_region_effect[region_id];

        // 检查是否满足分身条件（id >= 5 且 card_type_count >= 4）
        if region_id < 5 || !self.deck_can_split {
            return Ok(());
        }

        let clone_trains = &region.at_trains;
        if clone_trains.is_empty() {
            return Ok(());
        }

        // 获取所有可作为地区分身来源的支援卡（不含友人卡）
        //
        // 按 PersonType 扫全体人头，不写死 `0..6`。注意这在当前布局下是**防御性加固而非
        // bug 修复**：`PersonType::Card` 只来自 card_type 0..=4，恒占最低下标且至多 5 个，
        // 故 `(0..6).filter(Card)` 与全扫结果相同。写死上界只是等着下次人头布局变动时炸。
        let card_indices: Vec<i32> = (0..self.persons.len() as i32)
            .filter(|&i| self.persons[i as usize].person_type == PersonType::Card)
            .collect();
        if card_indices.is_empty() {
            return Ok(());
        }

        // 分身分配走局部流，与超级拉面用不同的 tag。
        //
        // 这条路径最吃「按 (rule_master, turn) 派生」的好处：地区分身在吃面落地时执行，
        // 父流 counter 取决于本回合此前的动作与事件，从父流 fork 会让上游任何位移
        // 都改掉选卡；按 (rule_master, turn, TAG) 派生则与上游完全无关。
        let mut clone_rng = self
            .clone_stream(CLONE_REGION_TAG)
            .unwrap_or_else(|| fork_local_stream(rng, CLONE_REGION_TAG));

        // 缺席优先：本回合被判定「不在」的支援卡（`PersonType::Card`，不含友人/理事长/记者）
        // 优先补进分身位——分身位顺序对应 `at_trains` 顺序，缺席卡按缺席记录顺序
        // 先缺谁补谁（先尝试第一个分身位，放不下再顺延）。直到全部支援卡都在训练后，
        // 剩余分身位才随机复制在场支援卡。
        let absent_cards: Vec<i32> = self
            .ramen
            .absent_cards
            .iter()
            .copied()
            .filter(|&i| self.persons[i as usize].person_type == PersonType::Card)
            .collect();
        let mut absent_placed = vec![false; absent_cards.len()];

        // 对于 at_trains 中的每个训练位置，随机选择一个不重复的支援卡分配分身
        //
        // 语义是 per-训练位（不是 per-卡）：某个位置放不下就是这个位置不出分身，
        // **不会**改去别的位置——那是超级拉面的语义。
        for &train in clone_trains {
            let train = train as usize;
            if train >= 5 {
                continue;
            }

            // 1) 缺席优先：按缺席记录顺序取第一个未被安置、且该位置合法的缺席卡
            let absent_pick = absent_cards
                .iter()
                .enumerate()
                .find(|&(k, &idx)| !absent_placed[k] && RamenAction::can_place_clone(self, idx, train))
                .map(|(k, &idx)| {
                    absent_placed[k] = true;
                    idx
                });
            if let Some(person_idx) = absent_pick {
                RamenAction::place_clone(self, person_idx, train, "地区(缺席优先)")?;
                continue;
            }

            // 2) 无缺席卡可用（全部在训练 / 放不下）：原逻辑随机复制在场支援卡
            // 先过滤合法卡再抽。原实现是「先抽再查满员」，满员时白白消耗一次随机数，
            // 且看起来像是算法失败，实际是规格内的跳过。
            let legal: Vec<i32> = card_indices
                .iter()
                .copied()
                .filter(|&idx| RamenAction::can_place_clone(self, idx, train))
                .collect();

            match legal.choose(&mut clone_rng) {
                Some(&person_idx) => RamenAction::place_clone(self, person_idx, train, "地区")?,
                None => {
                    // 规格内跳过：该位置满 5 个非 NPC，或全部支援卡都已在该位置。
                    crate::diag!(
                        ">> 地区分身跳过: {}训练无合法支援卡（满员或该位置已有全部支援卡）",
                        global!(GAMECONSTANTS).train_names[train]
                    );
                }
            }
        }

        Ok(())
    }

    /// 计算当前回合 hint_special 是否生效
    ///
    /// 生效条件：
    /// - 当前回合吃了面（current_ramen.is_some()）
    /// - ramen_basic_effect[year_idx].hint_special == true（仅第3年为 true）
    /// - 支援卡种类 >= 4（card_type_count >= 4）
    ///
    /// 超级拉面期间虽然 basic.hint_special 也生效，但此时不进行 hint 判定（直接享受 final 效果），
    /// 故此处判断为 false（不吃面时通过 current_ramen 短路掉即可）。
    fn calc_hint_special_active(&self, ramen: Option<usize>) -> bool {
        if ramen.is_none() {
            return false;
        }
        let ramen_data = global!(RAMENDATA);
        let year_idx = (self.current_year() - 1) as usize;
        if year_idx >= ramen_data.ramen_basic_effect.len() {
            return false;
        }
        if !ramen_data.ramen_basic_effect[year_idx].hint_special {
            return false;
        }
        // 支援卡种类 >= 4
        self.card_type_count.iter().filter(|&&x| x > 0).count() >= 4
    }

    /// 计算当前回合 hint_special 生效的训练位置集合（地区拉面 at_trains）
    fn calc_hint_special_at_trains(&self, ramen: Option<usize>) -> &'static [i32] {
        let ramen_data = global!(RAMENDATA);
        if let Some(region_idx) = ramen {
            if let Some(region) = ramen_data.ramen_region_effect.get(region_idx) {
                return &region.at_trains;
            }
        }
        &[]
    }

    /// 判断 hint_special 是否对指定 train 生效
    ///
    /// 用于 `handle_hint_event` 中区分 hint_special 路径与常规路径：
    /// hint_special 生效需要同时满足全局条件（吃面 + 第3年 + 支援卡种类>=4）
    /// 以及该 train 在当前回合吃的地区拉面的 at_trains 列表中。
    pub fn is_hint_special_active_for_train(&self, train: usize) -> bool {
        self.is_hint_special_active_for_train_with_ramen(train, self.ramen.current_ramen)
    }

    /// 按指定候选面判断训练位的隐藏 Hint，复用真实吃面状态的年度、卡种和覆盖规则。
    pub fn is_hint_special_active_for_train_with_ramen(&self, train: usize, ramen: Option<usize>) -> bool {
        if !self.calc_hint_special_active(ramen) {
            return false;
        }
        let at_trains = self.calc_hint_special_at_trains(ramen);
        at_trains.contains(&(train as i32))
    }
}

// ========== 私有辅助方法 ==========

impl RamenGame {
    // ========== 合并决策接口（仅 RamenGame，不放 Game trait） ==========

    /// 合并决策候选列表：不吃面 + 每个面 × `list_special_targets_for` 候选 targets
    ///
    /// 是 [`super::action::list_combined_ramen_select_actions`] 在 `RamenGame` 上的便捷转发。
    /// 适用于 MctsTrainer / 在线搜索等需要"选面+选吃法"一次性决策的场景。
    ///
    /// 与 `Game::list_actions` 的区别：
    /// - `Game::list_actions` 按当前 stage 分发（三阶段路径下 RamenSelect 只返回面选择）
    /// - 本方法直接在 RamenSelect 阶段返回 ramen × targets 笛卡尔积
    pub fn list_combined_ramen_select_actions(&self) -> Vec<super::action::RamenAction> {
        super::action::list_combined_ramen_select_actions(&self.ramen, &self.ramen.selected_regions)
    }

    /// 应用合并决策：在 RamenSelect 阶段一次性给出 ramen + targets 决策
    ///
    /// 与标准三阶段路径不同：调用本方法后 `Game::next()` 会直接把 stage 推到 Train，
    /// 跳过 SpecialSelect（靠 `RamenState::combined_decision` 标记位判断）。
    ///
    /// # 参数
    /// - `ramen`：选面决策；`None` 表示不吃面（此时 `targets` 被强制为 `[0,0,0]`）
    /// - `targets`：隐藏风味替换目标；吃面时必须在 `list_special_targets_for` 给出的
    ///   合法 targets 列表中，否则报错
    ///
    /// # 行为
    /// 1. 校验 stage 与 targets 合法性
    /// 2. 写 `pending_ramen` + `pending_special_targets`
    /// 3. 设 `combined_decision = true`
    /// 4. **不直接设 stage**，交给 `Game::next()` 推进（避免后续 next 混乱）
    ///
    /// 必须在 `stage == RamenStage::RamenSelect` 时调用；其他阶段调用返回错误。
    pub fn apply_combined_ramen_decision(&mut self, ramen: Option<usize>, targets: [i32; 3]) -> Result<()> {
        if self.stage != RamenStage::RamenSelect {
            anyhow::bail!(
                "apply_combined_ramen_decision: 仅在 RamenSelect 阶段可调用，当前 stage={:?}",
                self.stage
            );
        }

        // 不吃面强制 targets 全零
        let targets = match ramen {
            None => [0, 0, 0],
            Some(idx) => {
                // 校验 targets 是否合法
                let legal = super::rules::list_special_targets_for(&self.ramen, idx)?;
                if !legal.contains(&targets) {
                    anyhow::bail!(
                        "apply_combined_ramen_decision: targets {:?} 不在面 {} 的合法 targets 列表 {:?}",
                        targets,
                        idx,
                        legal
                    );
                }
                targets
            }
        };

        self.ramen.pending_ramen = ramen;
        self.ramen.pending_special_targets = targets;
        self.ramen.combined_decision = true;
        Ok(())
    }

    /// 推进到下一回合
    fn advance_turn(&mut self) -> bool {
        if self.base.turn < self.max_turn() {
            self.base.turn += 1;
            self.stage = RamenStage::Begin;
            if !self.check_free_race() {
                return false;
            }
            true
        } else {
            false
        }
    }

    /// 应用超级拉面的**一次性**赛后加成（`finals_effect.base.saihou` → `race_bonus`）。
    ///
    /// **幂等**：条件满足时合计最多生效一次（`super_ramen_saihou_applied` 置位后不再加），
    /// 故「模拟路径（[`Self::run_begin`]）」与「重放路径（`into_game`）」同时调用也不会重复加。
    ///
    /// 生效条件：`turn >= 72`（已进入超级拉面段）且已选超级拉面档位（`ramen.super_ramen` 为 `Some`）。
    /// 返回本次是否真正生效（供调用方决定日志）。
    ///
    /// 背景：协议帧不含 `raceBonus`，重放时 `parse_basegame` 只从支援卡累计，
    /// 缺了这 +100 会让 URA 段（turn 73/75/77 三次比赛）收益被系统性低估（×1.55 而非 ×2.55）。
    pub fn apply_super_ramen_saihou(&mut self) -> bool {
        if self.super_ramen_saihou_applied {
            return false;
        }
        if self.base.turn < 72 || self.ramen.super_ramen.is_none() {
            return false;
        }
        let saihou = global!(RAMENDATA).finals_effect.base.saihou;
        self.uma.race_bonus += saihou;
        self.super_ramen_saihou_applied = true;
        diag!(
            "超级拉面赛后加成（一次性）: race_bonus +{} → {}",
            saihou,
            self.uma.race_bonus
        );
        true
    }

    /// Begin 阶段：动态人头管理、隐藏风味、事件处理
    ///
    /// turn 2 只跑前半段，地区选择交给独立的 `RegionSelect` 阶段；
    /// 后半段由 [`Self::run_begin_suffix`] 在 `BeginAfterRegionSelect` 执行。
    /// 其它回合前半+后半连续跑完，行为与拆分前逐位一致。
    fn run_begin<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        self.run_begin_prefix()?;
        if self.base.turn == 2 {
            return Ok(());
        }
        self.run_begin_suffix(trainer, rng)
    }

    /// Begin 前半段：重置随机流、清 pending、日志、人头、诀窍初始化
    ///
    /// 每回合只应执行一次。`BeginAfterRegionSelect` **不得**再调用本函数。
    fn run_begin_prefix(&mut self) -> Result<()> {
        // 回合开始：重置两条规则流（注入 rule_master 后每回合从 0 计数，v2 §4.2）
        self.reset_turn_streams();
        // 三阶段决策 pending 防御性清空（Train 阶段结束后已清，但再确保一次）
        self.ramen.clear_pending();

        // 回合标题（turn_flow 风格分节；每回合一次）
        // 整段 cfg(feature = "diag")：MCTS rollout 编译时关掉 diag 可省
        // self.explain() 构造 String + self.explain_ramen_info() 构造 String。
        // 内层再包 enabled()：rollout 期间由 DiagGuard 静默（同上）。
        #[cfg(feature = "diag")]
        if crate::output::diagnostic::enabled() {
            diag!("────────── 回合 {} · 回合开始 ──────────", self.base.turn + 1);
            diag!("{}", self.explain()?);
            // 显示拉面杯信息（剧本机制未开启或URA回合时简化显示）
            let ramen_info = self.explain_ramen_info();
            if !ramen_info.is_empty() {
                diag!("{}", ramen_info);
            }
        }

        // 动态人头管理
        self.manage_persons_on_turn_start()?;

        // 诀窍值初始化/重置（回合2/24/48），同时处理隐藏风味
        // （init_feeling_stocks 内部已输出初始化结果，不重复打印回合信息）
        if matches!(self.base.turn, 2 | 24 | 48) {
            self.init_feeling_stocks();
        }
        Ok(())
    }

    /// Begin 后半段：隐藏风味、回合开始事件链、超级拉面自动效果
    ///
    /// turn 2 由 `BeginAfterRegionSelect` 调用；其它回合由 [`Self::run_begin`] 连续执行。
    /// **不得**重置随机流、加人头或初始化诀窍。
    fn run_begin_suffix<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let initialized = matches!(self.base.turn, 2 | 24 | 48);

        // 固定回合分配隐藏风味（初始化回合已由 init_feeling_stocks 处理，跳过）
        // （已输出"隐藏风味 +N"增量，不重复打印回合信息）
        if !initialized {
            let special = get_turn_special_feeling(self.base.turn);
            if special > 0 {
                self.ramen.special_feeling = (self.ramen.special_feeling + special).min(4);
                diag!("隐藏风味 +{} (={})", special, self.ramen.special_feeling);
            }
        }

        // ===== 事件流：回合开始事件链（v2 §4.3 三流第三轴）=====
        // 事件（马娘事件/支援卡连续事件）的随机独立于策略与局面——虽然是否触发
        // 依赖事件历史（`events` 计数 / max_time / 卡事件 8001-8003，策略状态），
        // 但随机序列本身与策略无关，故独立成 `event` 流：事件历史的差异只影响
        // 事件流自身，不污染局面流（角标/分布/hint，`run_distribute` 独占）与
        // 策略流（训练/分身/比赛）。
        let mut ev = self.event.take();
        // 休息心得结束判定（refresh_mind 由事件设置，随事件链走事件流）
        match ev.as_mut() {
            Some(s) => {
                if self.uma.flags.refresh_mind > 0 {
                    self.update_refresh_mind(s);
                }
            }
            None => {
                if self.uma.flags.refresh_mind > 0 {
                    self.update_refresh_mind(rng);
                }
            }
        }
        // 友人解锁判定（触发条件依赖 friend.out_state——是否点击友人）
        let unlock_event = match ev.as_mut() {
            Some(s) => self.try_friend_unlock(s),
            None => self.try_friend_unlock(rng)
        };
        // 事件生成（随机部分；Fixed 剧本事件无随机，天然逐位一致）
        let mut events = match ev.as_mut() {
            Some(s) => self.generate_events(s),
            None => self.generate_events(rng)
        };
        if unlock_event.is_some() {
            // 解锁触发时取代一般随机事件（与原语义一致）
            events = unlock_event.into_iter().collect();
        }
        self.add_mandatory_events(&mut events)?;
        // 事件应用（结果随机）
        for event in &events {
            match ev.as_mut() {
                Some(s) => self.run_event_on(event, trainer, rng, s)?,
                None => self.run_event(event, trainer, rng)?
            }
        }
        self.event = ev;

        // 超级拉面回合自动效果
        if self.is_super_ramen_turn() {
            if let Some(sel) = self.ramen.super_ramen {
                let options = rules::get_super_ramen_clone_train_options()?;
                if let Some(_option_trains) = options.get(sel) {
                    diag!("超级拉面回合自动生效 (选项 {})", sel + 1);
                }
            }
            // 应用 finals_effect.base 的 vital/motivation 恢复效果（每回合）
            // + saihou（赛后加成）一次性应用（幂等，见 `apply_super_ramen_saihou`）
            let ramen_data = global!(RAMENDATA);
            let finals_base = &ramen_data.finals_effect.base;
            let value = ActionValue {
                vital: finals_base.vital,
                motivation: finals_base.motivation,
                ..Default::default()
            };
            self.uma.add_value(&value);
            // 一次性赛后加成（race_bonus +saihou）：条件满足时合计只生效一次
            self.apply_super_ramen_saihou();
            diag!(
                "超级拉面自动恢复: 体力+{}, 干劲+{}",
                finals_base.vital,
                finals_base.motivation
            );
        }

        Ok(())
    }

    /// Distribute 阶段：分配人头和角标
    ///
    /// 随机来源：角标/人头分布/hint 走**回合固定流**（与策略无关，v2 §4.3）；
    /// 超级拉面分身分配走**策略流**（分身属策略交互随机）。
    /// 未注入 rule_master 时两者均回退旧行为（用传入 rng）。
    fn run_distribute(&mut self, rng: &mut impl Rng) -> Result<()> {
        if self.is_race_turn() {
            self.reset_distribution();
        } else {
            // 清零上一回合的「不在」记录（每位人头本回合只判一次）
            self.ramen.absent_cards.clear();
            // 回合固定流：角标 + 人头分布 + hint
            let mut fixed = self.turn_fixed.take();
            let raw_types = match fixed.as_mut() {
                Some(f) => assign_train_feeling_type(f),
                None => assign_train_feeling_type(rng)
            };
            self.turn_fixed = fixed;
            let feelings: [FeelingType; 5] =
                raw_types.map(|v| FeelingType::try_from(v).unwrap_or(FeelingType::A));
            // 超级拉面回合（72-77）角标**照常抽取但不落库**：诀窍机制已结束，
            // 训练不产生诀窍槽（与在线协议角标全 0 → None 一致）。不落库但保留
            // 抽签消耗，使后续 distribute_all/hint 的固定流偏移与旧行为逐位一致。
            self.ramen.train_feeling_type = if self.is_super_ramen_turn() { None } else { Some(feelings) };
            let mut fixed = self.turn_fixed.take();
            match fixed.as_mut() {
                Some(f) => {
                    self.distribute_all(f)?;
                    self.distribute_hint(f)?;
                }
                None => {
                    self.distribute_all(rng)?;
                    self.distribute_hint(rng)?;
                }
            }
            self.turn_fixed = fixed;

            // 超级拉面分身在 distribute_all 之后分配（策略流）
            if self.is_super_ramen_turn() {
                let mut strat = self.strategy.take();
                match strat.as_mut() {
                    Some(s) => super::action::RamenAction::distribute_super_ramen_clones(self, s)?,
                    None => super::action::RamenAction::distribute_super_ramen_clones(self, rng)?
                }
                self.strategy = strat;
            }

            // 训练后展示（comfy-table + ANSI 解析；MCTS rollout 编译时关 diag 跳过）
            // enabled() 包裹：rollout 期间跳过 explain_distribution 的构造（同上）
            #[cfg(feature = "diag")]
            if crate::output::diagnostic::enabled() {
                diag!("训练:\n{}", self.explain_distribution()?);
            }
        }
        Ok(())
    }

    /// Train 阶段：选择并执行动作
    ///
    /// Trainer 决策走决策流（`rng`）；动作执行（训练成败/休息/外出等）走**策略流**。
    fn run_train<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions()?;
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action_with_strategy(&actions[selection], rng)?;
        Ok(())
    }

    /// 用策略流执行动作（策略交互随机，v2 §4.3）
    ///
    /// 未注入 rule_master 时回退旧行为：用传入的决策 rng 执行。
    pub(crate) fn apply_action_with_strategy(&mut self, action: &RamenAction, rng: &mut StdRng) -> Result<()> {
        let mut strat = self.strategy.take();
        let result = match strat.as_mut() {
            Some(s) => self.apply_action(action, s),
            None => self.apply_action(action, rng)
        };
        self.strategy = strat;
        result
    }

    /// RamenSelect 阶段：选择吃哪碗面（含不吃）
    ///
    /// race_turn 时直接执行比赛，跳过 SpecialSelect/Train 阶段；
    /// 否则由 trainer 从候选面（不含/含至少一面）中选一个，apply 写 pending_ramen。
    fn run_ramen_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        // race_turn 短路：直接执行比赛，stage 切到 AfterTrain
        // 固定比赛回合仍先经过选面/隐藏风味阶段；Train 阶段只提供比赛动作。
        let actions = self.list_actions()?;
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action_with_strategy(&actions[selection], rng)?;
        // apply 已根据 ramen None/Some 自动切到 Train 或 SpecialSelect
        Ok(())
    }

    /// SpecialSelect 阶段：选择隐藏风味用法
    ///
    /// 由 trainer 从 `list_special_targets_for` 候选中选一个，apply 写 pending_special_targets。
    fn run_special_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions()?;
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action_with_strategy(&actions[selection], rng)?;
        // apply 已切到 Train
        Ok(())
    }

    /// AfterTrain 阶段：处理后续事件
    ///
    /// 事件结果随机走**策略流**（策略触发事件，v2 §4.3）；事件决策仍走决策流。
    fn run_after_train<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let mut after_events = std::mem::take(&mut self.base.unresolved_events);
        let mut strat = self.strategy.take();
        match strat.as_mut() {
            Some(s) => {
                for event in &after_events {
                    self.run_event_on(event, trainer, rng, s)?;
                }
            }
            None => {
                // 回退旧行为（未注入 rule_master）：与规则层改造前一致
                for event in &after_events {
                    self.run_event(event, trainer, rng)?;
                }
            }
        }
        self.strategy = strat;
        after_events.clear();
        self.base.unresolved_events = after_events;
        Ok(())
    }

    /// 事件执行：决策（player_select 选项）走 `decision_rng`，事件结果随机走 `rule_rng`
    //
    // 与 `Game::run_event` 默认实现（决策/规则共流）不同：规则层改造后事件结果
    // 必须由调用点决定用哪条规则流——回合开始固定事件用固定流，策略触发事件用策略流。
    fn run_event_on<T: Trainer<Self>>(
        &mut self, event: &EventData, trainer: &T, decision_rng: &mut StdRng, rule_rng: &mut impl Rng
    ) -> Result<()> {
        // 事件三段展示（标题 / 选项描述 / 选择结果）—— MCTS rollout 编译时关 diag 跳过
        // Explain::event_choice() 构造 String，开销可观
        // enabled() 包裹：rollout 期间由 DiagGuard 静默（同上）
        #[cfg(feature = "diag")]
        if crate::output::diagnostic::enabled() {
            diag!("【事件】#{} {}", event.id, event.name);
            if event.player_select && event.choices.len() > 1 {
                for (index, choice) in event.choices.iter().enumerate() {
                    diag!(
                        "  选项 {}: {}",
                        index + 1,
                        crate::explain::Explain::event_choice(choice)
                    );
                }
            }
        }
        let selection = if event.player_select && event.choices.len() > 1 {
            let selection = trainer.select_event_choice(self, event, &event.choices, decision_rng)?;
            if selection >= event.choices.len() {
                return Err(anyhow!(
                    "事件选项索引超出范围: selection={selection}, choices_len={}",
                    event.choices.len()
                ));
            }
            #[cfg(feature = "diag")]
            diag!("  → 选择 选项 {}", selection + 1);
            selection
        } else {
            0
        };
        if event.player_select && event.choices.len() > 1 {
            self.apply_event(&event, selection, rule_rng)
        } else {
            self.apply_event(&event, 0, rule_rng)
        }
    }

    /// 年度地区选择（通过 Trainer 统一接口决策）
    ///
    /// 候选由 [`list_actions`](Game::list_actions) 生成。第 3 年 `fixed` 策略
    /// **不经过 trainer** 直接落地（单候选直达，搜索侧 `list_actions` 也只给 1 个）。
    ///
    /// `year_idx`: 0=第1年(地区0-4), 1=第2年(地区5-9), 2=第3年(地区10-19)
    fn run_region_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng, year_idx: usize) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        let year = year_idx + 1;
        // 第 3 年 Fixed：跳过 trainer，避免 RandomTrainer 对单候选 shuffle 多消耗决策 rng。
        // 组合本身仍由 `region_select_combos` 这一份真值来源产出，避免与
        // `list_actions` 各读一次 GAMECONFIG 而在未来悄悄漂移。
        if year_idx == 2 && matches!(global!(GAMECONFIG).ramen_region_strategy, RamenRegionStrategy::Fixed) {
            let cfg = global!(GAMECONFIG);
            let combos = super::action::region_select_combos(
                year_idx,
                cfg.ramen_region_strategy,
                cfg.ramen_region_fixed.as_deref()
            )?;
            let combo = *combos
                .first()
                .ok_or_else(|| anyhow::anyhow!("第3年 Fixed 策略未产出任何组合"))?;
            let names: Vec<&str> = combo
                .iter()
                .filter_map(|&idx| ramen_data.ramen_region_effect.get(idx).map(|r| r.name.as_str()))
                .collect();
            diag!("==== 第3年 地区选择（Fixed 策略）: {} ====", names.join(", "));
            let action = RamenAction::no_ramen(Operation::RegionSelect(combo));
            self.apply_action_with_strategy(&action, rng)?;
            return Ok(());
        }
        let actions = self.list_actions()?;
        if actions.is_empty() {
            anyhow::bail!("RegionSelect 候选为空 (year_idx={year_idx}, turn={})", self.base.turn);
        }
        diag!("==== 第{}年 地区选择 ({}种组合) ====", year, actions.len());
        let selection = trainer.select_action(self, &actions, rng)?;
        if selection >= actions.len() {
            anyhow::bail!(
                "RegionSelect 选项索引超出范围: selection={selection}, actions_len={}",
                actions.len()
            );
        }
        self.apply_action_with_strategy(&actions[selection], rng)
    }

    /// SuperRamenSelect 阶段：由 trainer 从 `list_actions` 候选中选择
    ///
    /// 生产路径不再固定写入选项二；手写 / Local 基策在 trainer 侧查找
    /// `Operation::SuperRamenSelect(1)` 对应的候选位置，维持选项二回退。
    fn run_super_ramen_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions()?;
        if actions.is_empty() {
            anyhow::bail!("SuperRamenSelect 候选为空");
        }
        let selection = trainer.select_action(self, &actions, rng)?;
        if selection >= actions.len() {
            anyhow::bail!(
                "SuperRamenSelect 选项索引超出范围: selection={selection}, actions_len={}",
                actions.len()
            );
        }
        diag!("超级拉面选择: {}", actions[selection]);
        self.apply_action_with_strategy(&actions[selection], rng)?;
        Ok(())
    }

    /// 初始化/重置诀窍值和隐藏诀窍（回合 2/24/48 开始时）
    ///
    /// 根据携带的友人卡类型决定初始化数量：
    /// - 新友人(30305)：每种诀窍=2，隐藏诀窍+=2
    /// - 旧友人(9001/9008)：每种诀窍=1，隐藏诀窍+=1
    /// - 无友人卡：每种诀窍=0，隐藏诀窍+=1
    fn init_feeling_stocks(&mut self) {
        // 查找友人卡
        let friend_card = self.deck.iter().find(|c| c.card_type >= 5);
        let init_val = match friend_card {
            Some(card) if card.card_id == 30305 => 2,                     // 新友人
            Some(card) if matches!(card.data.chara_id, 9001 | 9008) => 1, // 旧友人
            _ => 0                                                        // 无友人卡
        };

        self.ramen.feeling_stock = [init_val; 3];
        // 无友人卡时仍获得1个隐藏风味
        let special_gain = if init_val > 0 { init_val } else { 1 };
        self.ramen.special_feeling = (self.ramen.special_feeling + special_gain).min(4);
        self.ramen.feeling_queue.clear();
        for _ in 0..init_val {
            for ft in [super::FeelingType::A, super::FeelingType::B, super::FeelingType::C] {
                self.ramen.feeling_queue.push(ft);
            }
        }
        diag!(
            ">> 诀窍初始化: 每种={} 隐藏+{} (={})",
            init_val,
            special_gain,
            self.ramen.special_feeling
        );
    }

    /// 更新休息心得
    ///
    /// 当 refresh_mind > 0 时，每回合开始时体力+5，并根据概率判定是否结束。
    fn update_refresh_mind(&mut self, rng: &mut impl Rng) {
        let t = self.uma.flags.refresh_mind as usize;
        if t > 0 {
            diag!("休息心得已持续 {t} 回合 -->");
            self.uma.add_value(&ActionValue { vital: 5, ..Default::default() });
            self.uma.flags.refresh_mind += 1;
            let end_prob = global!(GAMECONSTANTS).group_buff_end_prob[t.min(6)];
            if rng.random_bool(end_prob) {
                diag!(">> 休息心得结束");
                self.uma.flags.refresh_mind = 0;
            }
        }
    }

    /// 计算剧本 Hint 出现率加成百分比
    ///
    /// 来源：ramen_pt_effect.hint（常驻）+ ramen_success/fail_effect.hint（RMJ后）
    fn calc_hint_bonus_pct(&self) -> i32 {
        let ramen_data = global!(RAMENDATA);
        let year_idx = (self.current_year() - 1) as usize;

        // 1. ramen_pt_effect（常驻）
        let pt_tier = super::effects::find_pt_effect_tier(self.ramen.scenario_pt);
        let mut hint = ramen_data.ramen_pt_effect[pt_tier].hint;

        // 2. ramen_success/fail_effect（RMJ结算后）
        if year_idx >= 1 {
            let prev_idx = year_idx - 1;
            if let Some(&success) = self.ramen.rmj_results.get(prev_idx) {
                let rmj_effect = if success {
                    &ramen_data.ramen_success_effect[prev_idx]
                } else {
                    &ramen_data.ramen_fail_effect[prev_idx]
                };
                hint += rmj_effect.hint;
            }
        }
        hint
    }

    /// 动态人头管理：根据回合数添加友人卡、NPC和记者
    fn manage_persons_on_turn_start(&mut self) -> Result<()> {
        // 第2回合起：按需补齐友人卡与 5 个 NPC（各自独立判存在性，幂等）。
        //
        // **不能用**「是否已存在 ScenarioCard（友人卡人头）」当「已补过友人卡+NPC」
        // 的判据：协议层 `into_game` 重建 turn<2 帧时已把友人卡建成 ScenarioCard
        // 人头，用它会误判"已补齐"而整批跳过 NPC——链式推进出的 turn2 局面因此比
        // 真实帧少 5 个 NPC，训练值系统性偏低，运气分在 turn2 出现假跳升。
        if self.base.turn >= 2 {
            let mut added = false;
            if !self.persons.iter().any(|p| p.person_type == PersonType::ScenarioCard) {
                self.add_friend_card()?;
                added = true;
            }
            if !self.persons.iter().any(|p| p.person_type == PersonType::Npc) {
                self.add_npcs();
                added = true;
            }
            if added {
                diag!(">> 回合 {}：补人头，当前人头数 {}", self.base.turn, self.persons.len());
            }
        }
        // 第12回合起：按需添加记者（幂等）
        if self.base.turn >= 12 && !self.persons.iter().any(|p| p.person_type == PersonType::Reporter) {
            self.add_reporter();
            diag!(">> 回合 {}：添加记者，当前人头数 {}", self.base.turn, self.persons.len());
        }
        Ok(())
    }

    /// 格式化游戏状态（重写 BaseGame::explain，显示带剧本加成的训练等级）
    pub fn explain(&self) -> Result<String> {
        let mut lines = vec![];
        lines.push(format!(
            "回合: {}-{:?} 设施等级: {} 友人: {}",
            self.base.turn + 1,
            self.base.stage,
            crate::explain::Explain::train_level_count_with_bonus(
                &self.base.train_level_count,
                self.ramen.train_level_bonus
            ),
            self.base.friend.explain()
        ));
        // 体力警示已由 Uma::explain 的体力文字着色承担（<35 红、<50 黄）
        lines.push(self.base.uma.explain()?);
        Ok(lines.join("\n"))
    }

    /// 格式化拉面杯剧本信息（用于回合开始时显示）
    ///
    /// 包含：当前拉面地域及效果、当前选择地区、诀窍库存和槽值、剧本PT及加成档位
    /// 剧本机制未开启时（回合 < 2）返回空字符串
    /// URA回合（72-77）不显示地区、诀窍槽、诀窍点
    pub fn explain_ramen_info(&self) -> String {
        // 剧本机制未开启时，不显示拉面杯信息
        if self.base.turn < 2 {
            return String::new();
        }

        let ramen_data = global!(RAMENDATA);
        let is_ura = self.is_super_ramen_turn();

        // 当前拉面
        let ramen_str = if let Some(idx) = self.ramen.current_ramen {
            if let Some(region) = ramen_data.ramen_region_effect.get(idx) {
                // 计算地域效果生效的训练位置的效果
                let train = region.at_trains.first().copied().unwrap_or(0) as usize;
                let eff = super::effects::calc_ramen_training_effect(self, train, false);
                let mut parts = vec![];
                if eff.xunlian != 0 {
                    parts.push(format!("训+{}", eff.xunlian));
                }
                if eff.fail_rate_drop as i32 != 0 {
                    parts.push(format!("失败率-{}", eff.fail_rate_drop as i32));
                }
                if eff.friendship != 0 {
                    parts.push(format!("羁绊+{}", eff.friendship));
                }
                if eff.status_limit != 0 {
                    parts.push(format!("上限+{}", eff.status_limit));
                }
                if eff.pt_bonus != 0 {
                    parts.push(format!("PT+{}", eff.pt_bonus));
                }
                if parts.is_empty() {
                    region.name.clone()
                } else {
                    format!("{}({})", region.name, parts.join(","))
                }
            } else {
                "无".to_string()
            }
        } else {
            "无".to_string()
        };

        // URA回合：显示超级拉面加成
        if is_ura {
            let eff = super::effects::calc_ramen_training_effect(self, 0, false);
            let mut parts = vec![];
            if eff.xunlian != 0 {
                parts.push(format!("训+{}", eff.xunlian));
            }
            if eff.youqing != 0 {
                parts.push(format!("友情+{}", eff.youqing));
            }
            if eff.deyilv != 0 {
                parts.push(format!("得意+{}", eff.deyilv));
            }
            if eff.fail_rate_drop as i32 != 0 {
                parts.push(format!("失败率-{}", eff.fail_rate_drop as i32));
            }
            if eff.friendship != 0 {
                parts.push(format!("羁绊+{}", eff.friendship));
            }
            if eff.status_limit != 0 {
                parts.push(format!("上限+{}", eff.status_limit));
            }
            if eff.pt_bonus != 0 {
                parts.push(format!("PT+{}", eff.pt_bonus));
            }
            if eff.hint != 0 {
                parts.push(format!("hint+{}", eff.hint));
            }
            if eff.clone_count != 0 {
                parts.push(format!("分身+{}", eff.clone_count));
            }

            let mut result = format!("超级拉面回合");
            if !parts.is_empty() {
                result.push_str(&format!(" [{}]", parts.join(",")));
            }
            return result;
        }

        // 普通回合：完整显示
        // 当前选择地区
        let regions_str: Vec<String> = self
            .ramen
            .selected_regions
            .iter()
            .filter_map(|&idx| ramen_data.ramen_region_effect.get(idx).map(|r| r.name.clone()))
            .collect();

        // 诀窍库存和槽
        let stock = &self.ramen.feeling_stock;
        let slot = &self.ramen.feeling_slot;

        // 剧本PT加成档位
        let pt_tier = super::effects::find_pt_effect_tier(self.ramen.scenario_pt);
        let pt_effect = &ramen_data.ramen_pt_effect[pt_tier];
        let mut pt_parts = vec![];
        if pt_effect.xunlian != 0 {
            pt_parts.push(format!("训+{}", pt_effect.xunlian));
        }
        if pt_effect.deyilv != 0 {
            pt_parts.push(format!("得意+{}", pt_effect.deyilv));
        }
        if pt_effect.hint != 0 {
            pt_parts.push(format!("hint+{}", pt_effect.hint));
        }

        // 基础诀窍槽加成（并入"地区"栏显示）
        let base_dist = super::rules::calc_gauge_base_distribution(&self.ramen.selected_regions);

        // 诀窍 / 隐藏诀窍栏使用 cyan 突出显示
        let feeling_text = format!(
            "A{}/{} B{}/{} C{}/{}",
            stock[0], slot[0], stock[1], slot[1], stock[2], slot[2]
        );
        let special_text = self.ramen.special_feeling.to_string();

        format!(
            "拉面: {} | 地区: {} 槽{:?} | 诀窍 {} | 隐藏诀窍 {} | PT{} [{}]",
            ramen_str,
            regions_str.join(","),
            base_dist,
            feeling_text.cyan(),
            special_text.cyan(),
            self.ramen.scenario_pt,
            if pt_parts.is_empty() {
                "无加成".to_string()
            } else {
                pt_parts.join(",")
            }
        )
    }

    /// 添加强制事件（友人新年事件）
    ///
    /// 仅同步处理回合**开始时**发生的事件（push 到 `events`，立即 `run_event`）：
    /// - `turn=24` 友人新年事件（友人解锁后才有）
    ///
    /// 回合**结束时**发生的事件改由本函数内部直接 push 到 `base.unresolved_events`，
    /// 由 AfterTrain 阶段执行：
    /// - `turn=48` 新年抽签 4011（`system_events["ticket"]`，按 prob 加权选 result 分支）
    /// - `turn=77` 友人结束事件 + 育成结束事件 5011 + 401407
    ///
    /// 注：友人结束事件原本 push 到 `events` 在 Begin 阶段立即执行，但用户需求是"育成结束时（77 回合末尾）"
    /// 触发，所以改为 push 到 `unresolved_events` 在 AfterTrain 阶段执行。
    fn add_mandatory_events(&mut self, events: &mut Vec<EventData>) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        if self.friend.out_state == FriendOutState::AfterUnlock {
            if self.base.turn == 24 {
                events.push(ramen_data.friend_events["newyear"].clone());
            } else if self.base.turn == 77 {
                // 77 回合末尾：友人结束事件
                self.base
                    .unresolved_events
                    .push(ramen_data.friend_events["end"].clone());
            }
        }
        // 48 回合结束：新年抽签 4011
        if self.base.turn == 48 {
            self.base.unresolved_events.push(system_event("ticket")?.clone());
        }
        // 77 回合结束：育成结束事件 5011（ending）和 401407
        if self.base.turn == 77 {
            self.base
                .unresolved_events
                .push(system_event("ending").expect("ending event").clone());
            if let Some(event) = find_scenario_event(401407) {
                self.base.unresolved_events.push(event);
            }
        }
        Ok(())
    }

    /// 收集每回合训练数值 + 失败率 +（拉面回合）诀窍槽明细到 `lines`。
    ///
    /// 被 `explain_distribution` 在 cli / core 两种模式下复用，避免重复实现。
    /// 作为 inherent 方法（不属于 `Game` trait），保证 `Game::explain_distribution` 内
    /// 通过 `self.collect_train_lines(...)` 调用时优先匹配 inherent 实现。
    fn collect_train_lines(
        &self, lines: &mut Vec<String>, headers: &[String], _dist: &[Vec<i32>], show_ramen: bool
    ) -> Result<()> {
        for train in 0..5 {
            lines.push(self.train_value_line(&headers[train], train, show_ramen)?);
        }
        Ok(())
    }

    /// 单个训练位置的行级数值文本（`label + 数值 + 失败率 + 诀窍槽`）
    ///
    /// 计算逻辑与训练效果一致（buffs → 失败率 → 两阶段数值 → 诀窍槽明细），
    /// 供两处复用：分布表明细（label = 表头，如 `速C`）与 Train 候选预览
    /// （label = 训练名，如 `速训练`，见 [`Self::train_candidate_preview`]）。
    fn train_value_line(&self, label: &str, train: usize, show_ramen: bool) -> Result<String> {
        let buffs = self.calc_training_buff(train)?;
        let fail_rate = self.calc_training_failure_rate(&buffs, train);
        let base_value = self.calc_training_value(&buffs, train)?;
        let is_shining = self.shining_count(train) > 0;

        if !show_ramen {
            // 剧本机制未开启 或 URA回合：只显示基础训练数值和失败率
            return if fail_rate > 0.0 {
                Ok(format!("{label} {} 失败率: {}%", base_value.explain(), fail_rate))
            } else {
                Ok(format!("{label} {}", base_value.explain()))
            };
        }

        // 普通回合：训练数值（含拉面效果）+ 失败率 + 诀窍槽明细
        let ramen_effect = calc_ramen_training_effect(self, train, is_shining);
        let effective_fail = (fail_rate * (100.0 - ramen_effect.fail_rate_drop as f32) / 100.0)
            .min(100.0)
            .max(0.0);

        let value = ActionValue {
            status_pt: base_value.status_pt,
            vital: base_value.vital,
            motivation: base_value.motivation,
            ..Default::default()
        };

        let dist = &self.base.distribution;
        // 防护：`distribution` 未填满 5 行（早期回合 / unit-test 直接构造 game 调本方法），
        // 跳过该位的 buff 统计（不影响 explain_distribution 自身的 line 553 fill）
        if train >= dist.len() {
            // 训练位尚未就绪，返回一行只含失败率=0 + 数值零的占位文本，调用方仍可读
            return Ok(format!("{label} 训练位未就绪"));
        }
        let dist_train = &dist[train];
        let support_count = dist_train
            .iter()
            .filter(|&&p| {
                p >= 0
                    && (p as usize) < self.persons.len()
                    && self.persons[p as usize].person_type == crate::game::PersonType::Card
            })
            .count();
        // NPC 数量 = 本训练位置实际分配的 Npc 人数（`ramen_memo_cn.md` 算例：
        // NPC数量=3 时加成 floor(3/2)，非固定 5；与生效层 `fill_feeling_gauge` 一致）
        let npc_count = dist_train
            .iter()
            .filter(|&&p| {
                p >= 0
                    && (p as usize) < self.persons.len()
                    && self.persons[p as usize].person_type == crate::game::PersonType::Npc
            })
            .count();
        let train_feeling_bonus = super::rules::calc_train_feeling_bonus(support_count, npc_count);
        let base_dist = super::rules::calc_gauge_base_distribution(&self.ramen.selected_regions);
        let feeling_type = self.ramen.train_feeling_type.map(|types| types[train]);

        let gauge_a = base_dist[0]
            + if feeling_type == Some(super::FeelingType::A) {
                train_feeling_bonus
            } else {
                0
            }
            + if is_shining { 2 } else { 0 };
        let gauge_b = base_dist[1]
            + if feeling_type == Some(super::FeelingType::B) {
                train_feeling_bonus
            } else {
                0
            }
            + if is_shining { 2 } else { 0 };
        let gauge_c = base_dist[2]
            + if feeling_type == Some(super::FeelingType::C) {
                train_feeling_bonus
            } else {
                0
            }
            + if is_shining { 2 } else { 0 };

        let gauge_detail = format!("诀窍槽 A+{} B+{} C+{}", gauge_a, gauge_b, gauge_c);
        if effective_fail > 0.0 {
            Ok(format!(
                "{label} {} 失败率: {}% {}",
                value.explain(),
                effective_fail,
                gauge_detail
            ))
        } else {
            Ok(format!("{label} {} {}", value.explain(), gauge_detail))
        }
    }

    /// Train 阶段候选的内联预览文本（训练数值 + 失败率 + 诀窍槽）
    ///
    /// label 用训练名（`速训练`），供 RecordingTrainer / ramen_manual 把数值
    /// 内联到候选列表（如 `速训练 速17 力2 9pt 体力-22 诀窍槽 A+4 B+3 C+5`）。
    pub fn train_candidate_preview(&self, train: usize) -> Result<String> {
        let train_name = format!("{}训练", global!(GAMECONSTANTS).train_names[train]);
        let show_ramen = self.base.turn >= 2 && !self.is_super_ramen_turn();
        self.train_value_line(&train_name, train, show_ramen)
    }

    /// RamenSelect 阶段候选的内联预览文本（吃面后的完整效果）
    ///
    /// 效果口径与吃面落地后 `explain_ramen_info` 一致：克隆状态临时设置
    /// `current_ramen`，在地区 `at_trains` 首个位置计算 `calc_ramen_training_effect`
    /// （含 PT 常驻 + RMJ 常驻 + 基础效果 + 地区加成），输出如
    /// `吃面/中山-全 (训+20,友情+45,得意+140,失败率-50,上限+20,PT+5,hint+70)`。
    /// 基础效果与地区加成均包含在内；`is_shining=true` 保证地区/基础的友情
    /// 加成不被非友情训练归零（友情加成是吃面效果的一部分，玩家在选择时可见）。
    pub fn ramen_candidate_preview(&self, region_idx: usize) -> Result<String> {
        let ramen_data = global!(RAMENDATA);
        let region = ramen_data
            .ramen_region_effect
            .get(region_idx)
            .ok_or_else(|| anyhow::anyhow!("面索引 {region_idx} 不存在"))?;
        let mut preview = self.clone();
        preview.ramen.current_ramen = Some(region_idx);
        let train = region.at_trains.first().copied().unwrap_or(0) as usize;
        let eff = super::effects::calc_ramen_training_effect(&preview, train, true);
        let parts = super::effects::format_ramen_effect_parts(&eff);
        let name = region.name.clone();
        if parts.is_empty() {
            Ok(format!("吃面/{name}"))
        } else {
            Ok(format!("吃面/{name} ({})", parts.join(",")))
        }
    }
}

/// 按年份查找对应的 RMJ 事件（401404 / 401405 / 401406）
///
/// 返回事件 clone，供 push 到 `unresolved_events`。
/// 不存在时返回 None（数据缺失或年份越界）。
fn find_rmj_event(year_idx: usize) -> Option<crate::gamedata::EventData> {
    let ramen_data = global!(RAMENDATA);
    let target_id = match year_idx {
        0 => 401404,
        1 => 401405,
        2 => 401406,
        _ => return None
    };
    ramen_data.scenario_events.iter().find(|e| e.id == target_id).cloned()
}

/// 按 ID 在 scenario_events 中查找事件
///
/// 用于 push 未在 `add_mandatory_events` 处理的事件（如育成结束事件 401407）。
fn find_scenario_event(target_id: u32) -> Option<crate::gamedata::EventData> {
    let ramen_data = global!(RAMENDATA);
    ramen_data.scenario_events.iter().find(|e| e.id == target_id).cloned()
}

/// 判断事件 ID 是否为 RMJ 结算事件，若是则返回对应的年份索引（0/1/2）
///
/// 成功/失败分支选择见 `select_rmj_choice_by_result`。
fn rmj_event_year(event_id: u32) -> Option<usize> {
    match event_id {
        401404 => Some(0),
        401405 => Some(1),
        401406 => Some(2),
        _ => None
    }
}

/// 按 RMJ 结算结果（success=true/false）选择对应 result 分支
///
/// - `choices` 通常是 RMJ 事件的 `choices[0]`（选项组），含 2 个分支：
///   - `result=2`：成功（含大成功）
///   - `result=1`：失败
/// - `is_success`：来自 `rmj_results[year_idx]`，true 表示 result=2 分支，false 表示 result=1 分支
///
/// 选择规则：
/// - 优先按 `result` 字段匹配（成功→2，失败→1）
/// - 若无 `result` 字段匹配，则回退到第 0 个分支（防御性）
fn select_rmj_choice_by_result(
    choices: &[crate::gamedata::EventChoice], is_success: Option<bool>
) -> Option<&crate::gamedata::EventChoice> {
    if choices.is_empty() {
        return None;
    }
    let target_result = match is_success {
        Some(true) => 2,                 // 成功 → result=2
        Some(false) => 1,                // 失败 → result=1
        None => return Some(&choices[0]) // 无结算结果时回退到第一个分支
    };
    choices.iter().find(|c| c.result == target_result).or(Some(&choices[0]))
}

// ========== 测试 ==========

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::{
        game::{
            CardTrainingEffect, PersonType,
            ramen::events::assign_train_feeling_type,
            traits::{Game, Trainer}
        },
        gamedata::{ActionValue, EventChoice, init_global},
        trainer::{ManualTrainer, RandomTrainer},
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    /// 校验结果标记：`OK` / `NG`
    ///
    /// 项目约定测试用 `println` 输出而非 `assert`，故把判定结果打成可扫读的前缀。
    /// 回归失效时看输出里的 `NG` 即可定位，不中断后续用例的诊断信息。
    fn check(ok: bool) -> &'static str {
        if ok { "OK" } else { "NG" }
    }

    // 测试用公共参数
    // [速]杏目, [智]青春永驻, [耐]名将怒涛, [速]洛林军歌, [速]里见光钻, [友]骏川手纲
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };
    const TEST_UMA_ID: u32 = 102601;

    #[test]
    fn test_ramen_game_newgame() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        println!("开局人头数: {}", game.persons.len());
        println!("{}", game.explain()?);

        let card_count = game
            .persons
            .iter()
            .filter(|p| p.person_type == PersonType::Card)
            .count();
        let yayoi_count = game
            .persons
            .iter()
            .filter(|p| p.person_type == PersonType::Yayoi)
            .count();
        let npc_count = game.persons.iter().filter(|p| p.person_type == PersonType::Npc).count();
        let reporter_count = game
            .persons
            .iter()
            .filter(|p| p.person_type == PersonType::Reporter)
            .count();
        let scenario_count = game
            .persons
            .iter()
            .filter(|p| p.person_type == PersonType::ScenarioCard)
            .count();

        println!(
            "支援卡: {}, 理事长: {}, NPC: {}, 记者: {}, 友人卡: {}",
            card_count, yayoi_count, npc_count, reporter_count, scenario_count
        );

        assert_eq!(yayoi_count, 1, "开局应该有1个理事长");
        assert_eq!(npc_count, 0, "开局不应该有NPC");
        assert_eq!(reporter_count, 0, "开局不应该有记者");
        assert_eq!(scenario_count, 0, "开局不应该有友人卡");

        Ok(())
    }

    /// 确定性 RNG：`random::<f64>()` 恒为 0（`next_u64` 置 1、右移后为 0），
/// 故 `random_bool(p > 0)` 恒为 true。
///
/// 注意 `next_u64` **不能**返回 0：rand 0.9 的均匀整数采样
/// （Canon rejection，`lo >= thresh` 才接受）在恒 0 输入下会无限重试
/// （`range=500` 时 `thresh=116`）。返回 1 使 `wmul(500)` 的 `lo=500` 一次通过，
/// 且 `WeightedIndex` 仍恒选下标 0（`chosen_weight=0`）。
struct AlwaysTrueRng;
    impl rand::RngCore for AlwaysTrueRng {
        fn next_u32(&mut self) -> u32 {
            1
        }
        fn next_u64(&mut self) -> u64 {
            1
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(1);
        }
    }

    /// 不在权重类型表：支援卡 50、友人/团队卡 100、理事长/记者固定 200（不受
    /// `absent_rate_drop` 影响）、NPC 0（必定出现）
    #[test]
    fn test_absent_weight_by_type() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();
        let mut c = Checks::new();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.add_friend_and_npcs()?;
        // persons: 0-4 训练卡(Card)、5 理事长(Yayoi)、6 友人卡(ScenarioCard)、7-11 NPC
        let card_i = 0;
        let yayoi_i = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Yayoi)
            .expect("开局应有理事长");
        let friend_i = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::ScenarioCard)
            .expect("回合 2 应有友人卡");
        let npc_i = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Npc)
            .expect("回合 2 应有 NPC");

        c.check(game.absent_weight(card_i) == 50, "支援卡不在权重 = 50");
        c.check(game.absent_weight(yayoi_i) == 200, "理事长不在权重 = 200（固定）");
        c.check(game.absent_weight(friend_i) == 100, "友人卡不在权重 = 100");
        c.check(game.absent_weight(npc_i) == 0, "NPC 不在权重 = 0（必定出现）");
        c.finish()
    }

    /// 两步算法行为：不在判定与位置分配解耦（得意率只影响第二步）
    ///
    /// 用 `random_bool` 恒 true 的 RNG 验证——
    /// - 支援卡（不在权重 50）判定不在并记录，不进入位置分配；
    /// - NPC（权重 0）即使 RNG 恒 true 也必定出现；
    /// - `allow_absent=false`（追加分配）必定出现。
    #[test]
    fn test_distribute_person_two_stage_absent() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();
        let mut c = Checks::new();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.add_friend_and_npcs()?;
        game.reset_distribution(); // 单独调 distribute_person 前初始化 5 个空训练位
        let card_i = 0i32;
        let card2_i = 1i32;
        let npc_i = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Npc)
            .expect("回合 2 应有 NPC") as i32;

        let mut rng = AlwaysTrueRng;
        // 支援卡：random_bool 恒 true → 判定不在，绝不走到位置分配
        let r = game.distribute_person(card_i, true, &mut rng)?;
        c.check(r == -1, "支援卡在 random_bool 恒 true 时判定不在 (-1)");
        c.check(game.ramen.absent_cards == vec![card_i], "不在的支援卡被记录");
        c.check(
            game.distribution().iter().all(|d| d.is_empty()),
            "不在判定后未做位置分配（先判不在后分配）"
        );

        // NPC：不在权重 0 → 跳过不在判定，直接分配
        let r2 = game.distribute_person(npc_i, true, &mut rng)?;
        c.check(r2 >= 0, "NPC 在 random_bool 恒 true 时仍出现（无不在率）");

        // allow_absent=false：追加分配必定出现（跳过不在判定）
        let r3 = game.distribute_person(card2_i, false, &mut rng)?;
        c.check(r3 >= 0, "追加分配(allow_absent=false)在 random_bool 恒 true 时仍出现");
        c.finish()
    }

    /// 集成：多轮 `distribute_all` 下，不在记录只含支援卡/友人/团队卡、
    /// 与分布互斥，且 NPC 从未缺席
    #[test]
    fn test_absent_recorded_and_npc_always_present() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();
        let mut c = Checks::new();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.add_friend_and_npcs()?;
        // 记者在回合 12 才加入，这里直接塞一个记者人头验证其不在记录排除
        game.add_person(BasePerson {
            person_index: 0,
            person_type: PersonType::Reporter,
            train_type: -1,
            chara_id: 0,
            friendship: 0,
            is_hint: false,
            card_id: None
        });
        let npc_idx: Vec<i32> = game
            .persons
            .iter()
            .enumerate()
            .filter(|(_, p)| p.person_type == PersonType::Npc)
            .map(|(i, _)| i as i32)
            .collect();
        let mut rng = StdRng::seed_from_u64(20260825);
        let mut seen_absent = false;
        let mut seen_yayoi_absent = false;
        for round in 0..24 {
            game.ramen.absent_cards.clear(); // 模拟 run_distribute 的回合清理
            game.distribute_all(&mut rng)?;
            // NPC 必定出现
            for &i in &npc_idx {
                let present = game.at_trains(i).iter().any(|b| *b);
                c.check(present, &format!("第{round}轮 NPC#{i} 出现在训练中"));
            }
            // 不在记录：不含 NPC（必定出现）、与分布互斥；理事长/记者也一并记录
            for &i in &game.ramen.absent_cards {
                let ty = game.persons[i as usize].person_type;
                c.check(ty != PersonType::Npc, &format!("记录 #{i} 不含 NPC（实际 {ty:?}）"));
                let in_training = game.at_trains(i).iter().any(|b| *b);
                c.check(!in_training, &format!("记录 #{i} 当回合不出现在任何训练中"));
                if ty == PersonType::Yayoi {
                    seen_yayoi_absent = true;
                }
            }
            seen_absent |= !game.ramen.absent_cards.is_empty();
        }
        c.check(seen_absent, "24 轮分布中至少出现一次「不在」判定");
        c.check(seen_yayoi_absent, "理事长被判定不在时也入记录（剧本侧再按类型筛选）");
        c.finish()
    }

    /// 地区拉面分身——缺席优先：本回合判定「不在」的支援卡优先补进分身位
    ///
    /// 用 region 15（中山-速力智，`at_trains = [0, 2, 4]`），即用户示例的
    /// 「位置 [0,2,4] 需要分身」场景。
    #[test]
    fn test_region_clones_absent_priority() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();
        let mut c = Checks::new();

        let new_game = |absent: Vec<i32>| -> Result<(RamenGame, Vec<i32>)> {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.base.turn = 2;
            game.add_friend_and_npcs()?;
            game.reset_distribution();
            game.ramen.absent_cards = absent; // 模拟本回合 run_distribute 的判定结果
            let card_idx: Vec<i32> = (0..game.persons.len() as i32)
                .filter(|&i| game.persons[i as usize].person_type == PersonType::Card)
                .collect();
            Ok((game, card_idx))
        };

        // 场景 A：仅支援卡 1 缺席 → 优先出现在第一个分身位 0
        {
            let (mut game, _) = new_game(vec![1])?;
            let mut rng = StdRng::seed_from_u64(7);
            game.distribute_region_clones(15, &mut rng)?;
            let d = &game.base.distribution;
            println!("A(缺席[1]): 位置0={:?} 位置2={:?} 位置4={:?}", d[0], d[2], d[4]);
            c.check(d[0].contains(&1), "缺席支援卡 1 优先出现在第一个分身位 0");
            c.check(d[0] == vec![1], "位置 0 只有缺席卡 1 的分身（占位不追加随机复制）");
        }

        // 场景 B：支援卡 1、3 缺席 → 依次补 0、2 两个分身位
        {
            let (mut game, _) = new_game(vec![1, 3])?;
            let mut rng = StdRng::seed_from_u64(7);
            game.distribute_region_clones(15, &mut rng)?;
            let d = &game.base.distribution;
            println!("B(缺席[1,3]): 位置0={:?} 位置2={:?} 位置4={:?}", d[0], d[2], d[4]);
            c.check(d[0].contains(&1), "缺席卡 1 依次补位 → 位置 0");
            c.check(d[2].contains(&3), "缺席卡 3 依次补位 → 位置 2");
            c.check(!d[2].contains(&1), "缺席卡 1 不重复出现在位置 2");
        }

        // 场景 C：全员在训练（无缺席）→ 剩余分身位按原逻辑随机复制在场卡
        {
            let (mut game, cards) = new_game(vec![])?;
            let mut rng = StdRng::seed_from_u64(7);
            game.distribute_region_clones(15, &mut rng)?;
            let d = &game.base.distribution;
            println!("C(无缺席): 位置0={:?} 位置2={:?} 位置4={:?}", d[0], d[2], d[4]);
            for &t in &[0usize, 2, 4] {
                c.check(d[t].len() == 1, &format!("无缺席时位置 {t} 放一个复制分身"));
                if let Some(&x) = d[t].first() {
                    c.check(cards.contains(&x), &format!("位置 {t} 的分身来自支援卡 #{x}"));
                }
            }
        }
        c.finish()
    }

    /// 拉面杯要求卡组必须包含新友人卡（card_id=30305，rank 0-4，idrank 303050-303054）
    ///
    /// 校验逻辑：
    /// - 合法：idrank 满足 `idrank / 10 == 30305 && 0 <= rank <= 4`（rank=0 为未突破）
    /// - 非法：rank=5-9（303055-303059）、或完全无 30305
    #[test]
    fn test_ramen_newgame_requires_new_friend() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // 1. 不含新友人：应报错
        let deck_no_friend = [302424, 302894, 303044, 302924, 303024, 302924];
        let result = RamenGame::newgame(TEST_UMA_ID, &deck_no_friend, TEST_INHERIT);
        println!("无友人卡组: {:?}", result.is_err());
        assert!(result.is_err(), "卡组不含新友人应被拒绝");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("新友人"), "错误消息应提示新友人: {msg}");

        // 2. rank=0（idrank=303050，未突破）：应合法（旧注释曾误判为非法）
        let deck_rank0 = [302424, 302894, 303044, 302924, 303024, 303050];
        let result = RamenGame::newgame(TEST_UMA_ID, &deck_rank0, TEST_INHERIT);
        println!("rank=0 应合法: {}", result.is_ok());
        assert!(result.is_ok(), "rank=0（未突破）应合法");

        // 3. rank=5（idrank=303055）：应报错（rank 超出 [0,4]）
        let deck_rank5 = [302424, 302894, 303044, 302924, 303024, 303055];
        let result = RamenGame::newgame(TEST_UMA_ID, &deck_rank5, TEST_INHERIT);
        println!("rank=5 应被拒绝: {}", result.is_err());
        assert!(result.is_err(), "rank=5 应被拒绝（突破等级超出范围）");

        // 4. 合法 rank=1-4：应成功
        for rank in 1..=4u32 {
            let idrank = 303050 + rank;
            let deck = [302424, 302894, 303044, 302924, 303024, idrank];
            let result = RamenGame::newgame(TEST_UMA_ID, &deck, TEST_INHERIT);
            assert!(result.is_ok(), "rank={rank} (idrank={idrank}) 应合法");
        }

        Ok(())
    }

    #[test]
    fn test_ramen_game_full_loop() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;

        let trainer = RandomTrainer;
        let mut rng = StdRng::from_os_rng();
        println!("随机种子: {:?}", rng);

        println!("开始完整模拟...");
        game.run_full_game(&trainer, &mut rng)?;

        println!("育成结束!");
        println!("最终回合: {}", game.turn());
        println!("剧本PT: {}", game.ramen.scenario_pt);
        println!("RMJ结果: {:?}", game.ramen.rmj_results);
        println!("地区选择: {:?}", game.ramen.selected_regions);
        println!("超级拉面选择: {:?}", game.ramen.super_ramen);
        println!(
            "诀窍库存: A={} B={} C={}",
            game.ramen.feeling_stock[0], game.ramen.feeling_stock[1], game.ramen.feeling_stock[2]
        );
        println!("隐藏风味: {}", game.ramen.special_feeling);
        let score = game.uma.calc_score();
        println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);

        Ok(())
    }

    /// 静默测试游戏流程
    ///
    /// 关闭日志输出，仅输出育成配置和最终结果
    #[test]
    fn test_ramen_silent_loop() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error"); // 只输出错误
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let trainer = RandomTrainer;
        let mut rng = StdRng::from_os_rng();

        println!("=== 静默测试 ===");
        println!("卡组: {:?}", TEST_DECK);
        println!("随机种子: {:?}", rng);

        // 测试场景下不再 disable_log：cargo test 已隔离，
        // 日志输出到 stderr，按测试名天然不交错
        game.run_full_game(&trainer, &mut rng)?;

        // 输出最终结果
        println!("\n=== 育成结果 ===");
        println!("最终回合: {}", game.turn());
        println!("剧本PT: {}", game.ramen.scenario_pt);
        println!("RMJ结果: {:?}", game.ramen.rmj_results);
        println!("地区选择: {:?}", game.ramen.selected_regions);
        println!("超级拉面选择: {:?}", game.ramen.super_ramen);
        println!(
            "诀窍库存: A={} B={} C={}",
            game.ramen.feeling_stock[0], game.ramen.feeling_stock[1], game.ramen.feeling_stock[2]
        );
        println!("隐藏风味: {}", game.ramen.special_feeling);
        let score = game.uma.calc_score();
        println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);

        Ok(())
    }

    /// 回归：协议层重建的 turn<2 帧（友人卡已是 ScenarioCard 人头）推进到 turn2 时，
    /// 必须补齐 5 个 NPC。
    ///
    /// 旧守卫用「是否存在 ScenarioCard」判「是否已补过友人卡+NPC」：`into_game`
    /// 重建 turn<2 帧时已把友人卡建成 ScenarioCard，于是 turn2 整批跳过
    /// `add_friend_and_npcs` → 链式推进出的 turn2 局面少 5 个 NPC，训练值系统性
    /// 偏低（运气分 turn2 出现 ~+1300 假跳升）。
    #[test]
    fn test_turn2_backfills_npcs_when_friend_already_present() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 模拟 `into_game` 重建的 turn<2 帧：友人卡已作为 ScenarioCard 人头存在
        game.add_friend_card()?;
        let npc_before = game.persons.iter().filter(|p| p.person_type == PersonType::Npc).count();
        println!("turn<2 重建布局: persons={} npc={npc_before}", game.persons.len());

        game.base.turn = 2;
        game.manage_persons_on_turn_start()?;
        let npc_after = game.persons.iter().filter(|p| p.person_type == PersonType::Npc).count();
        println!("turn2 后: persons={} npc={npc_after}", game.persons.len());

        let mut c = Checks::new();
        c.check(npc_before == 0, "turn<2 无 NPC");
        c.check(npc_after == 5, "turn2 补齐 5 个 NPC");
        c.check(game.persons.len() == 12, "turn2 人头总数为 12");
        c.finish()
    }

    /// 训练参数分解日志专项测试
    ///
    /// 固定场景：回合 31（第二年，Lv=4），3 张速卡（杏目 id=0、洛林 id=3、里见 id=4）
    /// + 2 个 NPC 都在速训练位置，羁绊全部 100。然后分别在
    /// "不吃面"和"吃面 Some(5) 中京"两种情况下触发速训练，
    /// 输出 `explain_distribution` 和 `calc_train_params` 分解日志，
    /// 排查 issues.md「训练数值不对，尤其是友情加成」。
    #[test]
    fn test_train_param_decomposition() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 跳到回合 31（避开 102601 的生涯比赛回合）
        game.base.turn = 31;
        game.add_friend_and_npcs()?; // persons[0..5]=支援卡，[6]=友人卡，[7..12]=5个NPC
        game.add_reporter(); // persons[12]=记者
        // 所有支援卡羁绊 = 100（确保都能闪耀）
        for i in 0..6 {
            game.persons[i].friendship = 100;
            game.deck[i].friendship = 100;
        }
        // 第二年参数
        game.ramen.scenario_pt = 2000;
        game.ramen.rmj_results = vec![true]; // year 1 RMJ 成功
        // 训练次数全部 10，配合 train_level_bonus 让训练等级 = 4
        game.base.train_level_count = [10, 10, 10, 10, 10];
        game.ramen.train_level_bonus = 1;
        // 第 1 年地区选 [0, 6, 7]（札幌/中京/京都），便于 add_reporter 等流程
        game.ramen.selected_regions = [0, 6, 7];

        // 直接构造 distribution：3 张速卡 + 2 个 NPC 都在速训练位置
        game.base.distribution = vec![
            vec![0, 3, 4, 7, 8], // 速：杏目 + 洛林 + 里见 + NPC#1 + NPC#2
            vec![],              // 耐
            vec![],              // 力
            vec![],              // 根
            vec![],              // 智
        ];
        // 训练角标设为 [A, B, C, A, B]（无所谓，主要让 explain_distribution 不报错）
        game.ramen.train_feeling_type = Some([
            FeelingType::A,
            FeelingType::B,
            FeelingType::C,
            FeelingType::A,
            FeelingType::B
        ]);

        use crate::game::traits::{ActionEnum, Game};
        let mut rng = StdRng::seed_from_u64(42);

        // 跳到 Train 阶段
        game.stage = crate::game::ramen::RamenStage::Train;

        // ============ 场景 A：不吃面、速训练 ============
        game.ramen.current_ramen = None;
        let actions = game.list_actions()?;
        let train_idx = actions
            .iter()
            .position(|a| matches!(a.as_base_action(), Some(crate::game::BaseAction::Train(0))))
            .expect("应能找到速训练动作");
        println!("\n===== 场景 A：不吃面、速训练 =====\n{}", game.explain_distribution()?);
        game.apply_action(&actions[train_idx], &mut rng)?;

        // ============ 场景 B：吃面 Some(5) 中京、速训练 ============
        game.ramen.current_ramen = Some(5); // 中京 at_trains=[0,1,2,3,4], youqing=10
        let train_idx2 = actions
            .iter()
            .position(|a| matches!(a.as_base_action(), Some(crate::game::BaseAction::Train(0))))
            .expect("应能找到速训练动作");
        println!(
            "\n===== 场景 B：吃面 Some(5) 中京、速训练 =====\n{}",
            game.explain_distribution()?
        );
        game.apply_action(&actions[train_idx2], &mut rng)?;

        Ok(())
    }

    /// 训练后事件按入队顺序各处理一次，空队列不重放，并保留容量供下一回合使用。
    #[test]
    fn test_after_train_events_process_once_and_reuse_queue() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        init_global()?;
        let mut c = Checks::new();

        for master in [None, Some(61444)] {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            if let Some(master) = master {
                game.set_rule_master(master);
            }
            let mut rng = StdRng::seed_from_u64(42);
            game.base.unresolved_events = Vec::with_capacity(4);
            let capacity = game.base.unresolved_events.capacity();
            for round in 1..=2 {
                game.base.uma.vital = game.base.uma.max_vital - 5;
                game.base.unresolved_events.extend([20, -10].into_iter().enumerate().map(|(i, vital)| {
                    EventData {
                        id: 90000 + i as u32,
                        choices: vec![vec![EventChoice {
                            value: ActionValue { vital, ..Default::default() },
                            ..Default::default()
                        }]],
                        ..Default::default()
                    }
                }));
                game.run_after_train(&RandomTrainer, &mut rng)?;
                // 紧接着处理空队列，不能重复应用刚刚清掉的事件。
                game.run_after_train(&RandomTrainer, &mut rng)?;
                println!("规则种子={master:?}，轮次={round}，体力={}，队列容量={}",
                    game.uma.vital, game.base.unresolved_events.capacity());
                c.check(game.uma.vital == game.uma.max_vital - 10, "先恢复至上限再扣除体力，事件顺序不变");
                c.check(
                    [90000, 90001].iter().all(|id| game.base.events.get(id) == Some(&round)),
                    "两个事件每轮各生效一次，空队列不重放"
                );
                c.check(game.base.unresolved_events.is_empty(), "事件处理后队列为空");
                c.check(game.base.unresolved_events.capacity() == capacity, "后续回合复用事件队列容量");
                c.check(game.strategy.is_some() == master.is_some(), "事件处理后保留策略流");
            }
        }
        c.finish()
    }

    #[test]
    fn test_random_event_generation() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // 创建游戏实例
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;

        // 使用随机种子
        let mut rng = StdRng::from_os_rng();
        println!("随机种子: {:?}", rng);

        // 模拟一整年（24回合）的事件生成
        println!("\n========== 模拟一整年（24回合）的事件生成 ==========");
        let mut total_events = 0;
        let mut event_counts = std::collections::HashMap::new();

        for turn in 1..=24 {
            game.base.turn = turn;
            let events = game.generate_events(&mut rng);

            println!("\n回合 {}: 生成 {} 个事件", turn, events.len());
            for (i, event) in events.iter().enumerate() {
                println!("  事件 {}: ID={}, 名称={}", i + 1, event.id, event.name);
                total_events += 1;
                *event_counts.entry(event.name.clone()).or_insert(0) += 1;
                // 更新事件计数（模拟 apply_event 的计数逻辑）
                *game.base.events.entry(event.id).or_insert(0) += 1;
            }

            if events.is_empty() {
                println!("  无事件触发");
            }
        }

        // 输出统计信息
        println!("\n========== 事件统计 ==========");
        println!("总事件数: {}", total_events);
        println!("平均每次回合事件数: {:.2}", total_events as f64 / 24.0);

        println!("\n事件类型统计:");
        let mut sorted_events: Vec<_> = event_counts.iter().collect();
        sorted_events.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in sorted_events {
            println!("  {}: {} 次", name, count);
        }

        // 验证事件生成逻辑
        println!("\n========== 事件分布验证 ==========");
        let event_dist = global!(GAMECONSTANTS).get_event_distribution();
        println!("事件分布配置: {:?}", event_dist);
        println!("说明: [支援卡事件, 马娘事件, 掉心情事件, 无事件]");

        Ok(())
    }

    /// 端到端训练数值测试：固定回合 30（第二年），分别打印不吃面 / 吃面 Some(5) 的训练信息
    ///
    /// 固定场景：
    /// - 回合 30（第二年），友人和全部 NPC 已解锁，记者已加入
    /// - feeling_stocks = [3, 3, 3]，地区选择 [5, 6, 7]，scenario_pt = 3000
    /// - rmj_results = [true]（第 1 年 RMJ 成功），所有支援卡羁绊设为 100
    /// - 随机产生 1 次训练分配，分别以 `current_ramen = None` 和 `current_ramen = Some(5)`
    ///   复用同一份分配，调用 `explain_distribution` 输出训练信息
    ///
    /// 主要观测点：
    /// 1. `is_shining_at` 判定（闪耀标记）是否符合"得意位置 + 羁绊 ≥ 80"
    /// 2. 不吃面 vs 吃面：吃面是否引入了 `basic_effect`（羁绊/失败率等）和
    ///    命中 `at_trains` 的 `region_effect`（xunlian/youqing/pt_bonus）
    /// 3. 拉面杯加成的累乘结果是否与公式一致
    #[test]
    fn test_random_distribution_training_value() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // 1. 创建游戏并直接跳到回合 30（第二年）
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 30;
        // 2. 解锁友人和全部 NPC（person_is_available 要求 turn >= 2）
        game.add_friend_and_npcs()?;
        // 3. 加入记者（person_is_available 要求 turn >= 12）
        game.add_reporter();
        // 4. feeling_stocks = [3, 3, 3]
        game.ramen.feeling_stock = [3, 3, 3];
        // 5. 地区选择 [5, 6, 7]
        game.ramen.selected_regions = [5, 6, 7];
        // 6. scenario_pt = 3000
        game.ramen.scenario_pt = 3000;
        // 7. rmj_results = [true]（第 1 年 RMJ 成功 → 第 2 年常驻 ramen_success_effect[0]）
        game.ramen.rmj_results = vec![true];
        // 直接跳到回合 30 跳过了 RMJ 结算的 train_level_bonus += 1，
        // 这里手动 +1，使 Lv = 10/4 + 1 + 1 = 4
        game.ramen.train_level_bonus = 1;
        // 每个训练的点击次数设为 10，配合 bonus=1 使实际训练等级 = 4
        game.base.train_level_count = [10, 10, 10, 10, 10];
        // 8. 所有支援卡羁绊设为 100（顺手同步 persons / deck 两处）
        for i in 0..6 {
            game.persons[i].friendship = 100;
            game.deck[i].friendship = 100;
        }
        for p in game.persons.iter_mut() {
            if p.person_type == PersonType::Card {
                p.friendship = 100;
            }
        }

        let mut rng = StdRng::from_os_rng();
        println!("\n========== 端到端训练数值测试 ==========");
        println!("随机种子: {:?}", rng);

        // ========== 详细回合信息 ==========
        println!("\n----- 回合信息 -----");
        println!("回合: {} (第{}年)", game.base.turn, game.current_year());
        println!("地区选择: {:?}", game.ramen.selected_regions);
        println!(
            "地区词条: {}",
            game.ramen
                .selected_regions
                .iter()
                .map(|&i| {
                    let ramen_data = global!(RAMENDATA);
                    ramen_data.ramen_region_effect[i].name.clone()
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("剧本 PT: {}", game.ramen.scenario_pt);
        println!("RMJ 结果: {:?}", game.ramen.rmj_results);
        println!("训练等级加成: {}", game.ramen.train_level_bonus);
        println!("训练点击次数: {:?}", game.base.train_level_count);
        println!(
            "feeling_stocks: A={} B={} C={}",
            game.ramen.feeling_stock[0], game.ramen.feeling_stock[1], game.ramen.feeling_stock[2]
        );
        println!("隐藏风味: {}", game.ramen.special_feeling);
        println!("人头总数: {}", game.persons.len());

        // ========== 支援卡羁绊概览 ==========
        println!("\n----- 支援卡羁绊 -----");
        for i in 0..6 {
            let p = &game.persons[i];
            println!(
                "  [#{}] {} 类型={} 羁绊={}",
                i,
                p.short_name(),
                p.train_type,
                p.friendship
            );
        }

        // ========== 随机分配 1 次（两个场景共用同一份分配） ==========
        let raw_types = assign_train_feeling_type(&mut rng);
        let feelings: [FeelingType; 5] = raw_types.map(|v| FeelingType::try_from(v).unwrap_or(FeelingType::A));
        game.ramen.train_feeling_type = Some(feelings);
        game.distribute_all(&mut rng)?;
        game.distribute_hint(&mut rng)?;

        // ========== 场景1：不吃面 ==========
        game.ramen.current_ramen = None;
        println!("\n========== 场景1：current_ramen = None（不吃面）==========");
        println!(
            "训练等级: 速={} 耐={} 力={} 根={} 智={}",
            game.train_level(0),
            game.train_level(1),
            game.train_level(2),
            game.train_level(3),
            game.train_level(4)
        );
        println!("\n{}", game.explain_distribution()?);

        // ========== 场景2：吃面 Some(5) ==========
        game.ramen.current_ramen = Some(5);
        let ramen_data = global!(RAMENDATA);
        let region = &ramen_data.ramen_region_effect[5];
        println!(
            "\n========== 场景2：current_ramen = Some(5) ==========\n        地区 {} xunlian={} youqing={} pt_bonus={} hint_count={} at_trains={:?}",
            region.name, region.xunlian, region.youqing, region.pt_bonus, region.hint_count, region.at_trains
        );
        println!(
            "训练等级: 速={} 耐={} 力={} 根={} 智={}",
            game.train_level(0),
            game.train_level(1),
            game.train_level(2),
            game.train_level(3),
            game.train_level(4)
        );
        println!("\n{}", game.explain_distribution()?);

        Ok(())
    }

    /// 验证 RamenGame::deyilv 返回"卡 deyilv + 剧本 deyilv 总加成"
    ///
    /// 关键点：
    /// - 普通回合：剧本 deyilv = pt_effect(当前档) + rmj_results[year-1] success/fail
    /// - 超级拉面：剧本 deyilv = pt_effect(最后一档) + rmj_results[2] success/fail
    /// - 调用方拿到这个值后，会作为 distribute_person 的训练位置权重加成
    #[test]
    fn test_ramen_deyilv_includes_scenario_bonus() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // ========== 普通回合（year 2, PT=1000, RMJ 成功） ==========
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 30; // year 2
        game.add_friend_and_npcs()?; // person[0..4] 是训练卡，5 是理事长，6 是友人卡
        game.ramen.scenario_pt = 1000;
        game.ramen.rmj_results = vec![true]; // year 1 RMJ 成功

        // 卡 deyilv 来自 calc_training_effect，剧本 deyilv = pt(1000档=63) + rmj_success[0]=80 = 143
        let person_idx = 0;
        let card_deyilv_only = game.deck[person_idx].effect.deyilv;
        let actual_deyilv = game.deyilv(person_idx as i32);
        println!(
            "year2, PT=1000, RMJ成功: card_deyilv_only={} 实际 deyilv={}",
            card_deyilv_only, actual_deyilv
        );
        // 期望：actual_deyilv = card_deyilv_only + 143
        assert_eq!(actual_deyilv, card_deyilv_only + 143.0);

        // ========== 超级拉面（turn=72, PT=5000, RMJ 都成功） ==========
        let mut game2 = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game2.base.turn = 72;
        game2.add_friend_and_npcs()?;
        game2.ramen.scenario_pt = 5000;
        game2.ramen.rmj_results = vec![true, true, true];

        let card_deyilv_only2 = game2.deck[person_idx].effect.deyilv;
        let actual_deyilv2 = game2.deyilv(person_idx as i32);
        println!(
            "超级拉面, PT=5000, RMJ都成功: card_deyilv_only={} 实际 deyilv={}",
            card_deyilv_only2, actual_deyilv2
        );
        // 期望：actual_deyilv = card_deyilv_only + rmj_success[2]=250 = +250
        // （超级拉面不叠加 pt_effect 的 deyilv=80）
        assert_eq!(actual_deyilv2, card_deyilv_only2 + 250.0);

        // ========== 无卡人头返回 0，友人卡按自己的卡组槽位取值 ==========
        // 旧断言写死「person_index >= 6 返回 0」，但拉面布局下人头 6 正是友人卡。
        // 这里按 PersonType 定位，不依赖任何硬编码下标。
        let yayoi_idx = game2
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Yayoi)
            .ok_or_else(|| anyhow!("找不到理事长人头"))?;
        let yayoi_deyilv = game2.deyilv(yayoi_idx as i32);
        println!(
            "[{}] 理事长(人头 {yayoi_idx}) 无卡，deyilv 应为 0：实际 {yayoi_deyilv}",
            check(yayoi_deyilv == 0.0)
        );

        let friend_idx = game2
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::ScenarioCard)
            .ok_or_else(|| anyhow!("找不到友人卡人头"))?;
        let friend_deck_idx = Game::deck_index_of(&game2, friend_idx).ok_or_else(|| anyhow!("友人卡反查卡组失败"))?;
        let friend_card_deyilv = game2.deck[friend_deck_idx].effect.deyilv;
        let friend_deyilv = game2.deyilv(friend_idx as i32);
        println!(
            "[{}] 友人卡(人头 {friend_idx} -> 卡组 {friend_deck_idx}): 期望 {} 实际 {friend_deyilv}",
            check(friend_deyilv == friend_card_deyilv + 250.0),
            friend_card_deyilv + 250.0
        );

        // 负数人头下标不再 panic
        println!("deyilv(-1)={}", game2.deyilv(-1));

        Ok(())
    }

    /// 三阶段决策衔接测试
    ///
    /// 手动模拟回合 2 的 RamenSelect → SpecialSelect → Train 全流程，
    /// 验证阶段切换与 pending 字段在阶段间正确传递。
    #[test]
    fn test_three_stage_decision_flow() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 跳到回合 13：turn >= 2 才有吃面选择，turn > 12 才允许比赛
        game.base.turn = 13;
        // 直接给一个够库存的状态（手动跳过 RegionSelect 等阶段）
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2]; // 札幌、函馆、新潟

        // 把 stage 推进到 RamenSelect（手动 set，不经过真实流程）
        game.stage = RamenStage::RamenSelect;

        // ===== 阶段1：RamenSelect =====
        let actions = game.list_actions()?;
        println!("RamenSelect 阶段: {actions:#?}");
        assert!(actions.len() >= 1, "至少有'不吃面'候选");
        // 所有动作 operation 必须是 StageOnly
        for a in &actions {
            assert!(
                matches!(a.operation, Operation::StageOnly),
                "RamenSelect 阶段动作 operation 必须是 StageOnly"
            );
        }
        // 选第一个面（确保库存够）
        let pick_idx = actions
            .iter()
            .position(|a| a.ramen.is_some())
            .expect("至少有一个候选面");
        let ramen_idx = actions[pick_idx].ramen.expect("已 Some");
        game.apply_action(&actions[pick_idx], &mut StdRng::from_os_rng())?;

        // 验证 pending_ramen 已写
        assert_eq!(game.ramen.pending_ramen, Some(ramen_idx));
        println!("pending_ramen: {:?}", game.ramen.pending_ramen);
        // apply 不切 stage；外部 next() 决定推进
        assert!(matches!(game.stage, RamenStage::RamenSelect));

        // 推进 stage：模拟 Game::next() 行为
        let next_stage = if game.ramen.pending_ramen.is_some() {
            RamenStage::SpecialSelect
        } else {
            RamenStage::Train
        };
        game.stage = next_stage;

        // ===== 阶段2：SpecialSelect =====
        let actions = game.list_actions()?;
        println!("SpecialSelect 阶段: {actions:#?}");
        assert!(actions.len() >= 1, "至少有 1 个 targets 候选");
        for a in &actions {
            assert!(
                matches!(a.operation, Operation::StageOnly),
                "SpecialSelect 阶段动作 operation 必须是 StageOnly"
            );
            assert_eq!(a.ramen, Some(ramen_idx));
            assert!(
                a.special_targets.is_some(),
                "SpecialSelect 阶段动作应携带 special_targets"
            );
        }

        // 选第一个 targets（按 sum 升序通常第一个是最小必要）
        let chosen_targets = actions[0].special_targets.expect("已 Some");
        game.apply_action(&actions[0], &mut StdRng::from_os_rng())?;

        // 验证 pending_special_targets 已写
        println!("pending_special_targets: {:?}", game.ramen.pending_special_targets);
        assert_eq!(game.ramen.pending_special_targets, chosen_targets);

        // 推进 stage
        game.stage = RamenStage::Train;

        // ===== 阶段3：Train =====
        // 重构后：Train 阶段动作不再携带 ramen/special_targets 字段
        // （这两个字段已由 SpecialSelect → Train 过渡时的 ground_ramen_effects 落地）
        let actions = game.list_actions()?;
        println!("Train 阶段: {actions:#?}");
        assert!(actions.len() >= 8);
        for a in &actions {
            assert_eq!(a.ramen, None, "Train 阶段动作 ramen 应为空（已 ground）");
            assert_eq!(
                a.special_targets, None,
                "Train 阶段动作 special_targets 应为空（已 ground）"
            );
            assert!(
                !matches!(a.operation, Operation::StageOnly),
                "Train 阶段动作 operation 不应是 StageOnly"
            );
        }

        Ok(())
    }

    /// 合并决策路径端到端测试
    ///
    /// 验证：在 RamenSelect 阶段使用 `apply_combined_ramen_decision` 一次性给出
    /// ramen + targets 后，`Game::next()` 直接把 stage 推到 Train，跳过 SpecialSelect。
    /// 同时验证三阶段路径与合并路径在同一回合内互不干扰。
    #[test]
    fn test_combined_decision_path_skips_special_select() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];

        // 把 stage 推到 RamenSelect
        game.stage = RamenStage::RamenSelect;
        assert!(!game.ramen.combined_decision);

        // ===== 合并决策：选面 0 + targets=[1,0,0] =====
        let combined_actions = game.list_combined_ramen_select_actions();
        println!("合并决策候选数: {}", combined_actions.len());
        // 3 面全富余下：1(不吃) + 9(札幌) + 9(函馆) + 8(新潟) = 27
        assert!(
            combined_actions.len() >= 27,
            "3 面全富余应至少 27 个（实测 {}）",
            combined_actions.len()
        );

        let chosen = combined_actions
            .iter()
            .find(|a| a.ramen == Some(0) && a.special_targets == Some([1, 0, 0]))
            .copied()
            .expect("候选中应包含 面0 + [1,0,0]");

        // 应用合并决策
        game.apply_combined_ramen_decision(chosen.ramen, chosen.special_targets.unwrap())?;

        // 验证 pending 字段已写 + 标记位已设
        assert_eq!(game.ramen.pending_ramen, Some(0));
        assert_eq!(game.ramen.pending_special_targets, [1, 0, 0]);
        assert!(game.ramen.combined_decision, "combined_decision 应为 true");
        // stage 仍是 RamenSelect（不直接设 stage）
        assert!(matches!(game.stage, RamenStage::RamenSelect));

        // ===== Game::next() 推进：合并决策应直接推 Train，跳过 SpecialSelect =====
        game.next();
        println!("next() 后 stage: {:?}", game.stage);
        assert!(
            matches!(game.stage, RamenStage::Train),
            "合并决策路径应直接推 Train（跳过 SpecialSelect）"
        );

        // ===== 关键不变性：再次 next() 不应再推 SpecialSelect =====
        // （SpecialSelect 已被跳过；如果 next() 误推会出错）
        let prev_stage = game.stage.clone();
        // 不再调 next()（会推进到 AfterTrain）；只校验 stage 已是 Train

        // ===== clear_pending 后 combined_decision 应清空（回合边界语义） =====
        game.ramen.clear_pending();
        assert!(!game.ramen.combined_decision);
        assert_eq!(game.ramen.pending_ramen, None);
        assert_eq!(game.ramen.pending_special_targets, [0, 0, 0]);
        println!("clear_pending 后所有 pending 已清空（含 combined_decision）");

        // 防止 "unused" 警告
        let _ = prev_stage;

        Ok(())
    }

    /// 合并决策路径"不吃面"分支测试
    ///
    /// 验证 `apply_combined_ramen_decision(None, ...)` 强制 targets=[0,0,0] 且
    /// `Game::next()` 同样直接推 Train（与"三阶段不吃面"行为一致）。
    #[test]
    fn test_combined_decision_path_no_ramen() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;

        // 不吃面 + 任意 targets（应被强制成 [0,0,0]）
        game.apply_combined_ramen_decision(None, [2, 2, 2])?;
        assert_eq!(game.ramen.pending_ramen, None);
        assert_eq!(game.ramen.pending_special_targets, [0, 0, 0]);
        assert!(game.ramen.combined_decision);

        // next() 推到 Train
        game.next();
        assert!(matches!(game.stage, RamenStage::Train));

        Ok(())
    }

    /// 合并决策路径非法 targets 应报错
    #[test]
    fn test_combined_decision_invalid_targets_rejected() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;

        // 面 0 札幌 recipe=[2,2,1]，targets=[3,0,0] 不合法（t_a 超过 recipe[0]=2）
        let result = game.apply_combined_ramen_decision(Some(0), [3, 0, 0]);
        println!("非法 targets 应报错: {:?}", result.is_err());
        assert!(result.is_err(), "targets 越界应被拒绝");

        // pending 应未写入
        assert_eq!(game.ramen.pending_ramen, None);
        assert!(!game.ramen.combined_decision);

        Ok(())
    }

    /// 三阶段路径在 combined_decision=false 时行为不变（回归测试）
    ///
    /// 确认方案 E 不影响 HandwrittenTrainer 等走三阶段的 Trainer：
    /// RamenSelect → next() 仍按 pending_ramen 决定 SpecialSelect / Train。
    #[test]
    fn test_three_stage_path_unaffected_by_combined_flag() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;
        assert!(!game.ramen.combined_decision);

        // 走三阶段路径：选面 0 后 apply，写 pending_ramen
        let actions = game.list_actions()?;
        let pick = actions.iter().position(|a| a.ramen == Some(0)).expect("应有面 0 候选");
        game.apply_action(&actions[pick], &mut StdRng::from_os_rng())?;

        // combined_decision 应保持 false（apply_action 走中间步骤，不设标记）
        assert!(!game.ramen.combined_decision);
        assert_eq!(game.ramen.pending_ramen, Some(0));

        // next() 应推 SpecialSelect（标准三阶段路径）
        game.next();
        assert!(
            matches!(game.stage, RamenStage::SpecialSelect),
            "三阶段路径下 RamenSelect → SpecialSelect"
        );

        Ok(())
    }

    // ========== RMJ 结算事件 + 固定触发事件 测试 ==========

    /// 验证 `select_rmj_choice_by_result` 的分支选择逻辑
    #[test]
    fn test_select_rmj_choice_by_result() {
        let choices = vec![
            EventChoice {
                result: 2, // 成功
                value: ActionValue {
                    status_pt: [10, 10, 10, 10, 10, 100],
                    vital: 33,
                    ..Default::default()
                },
                ..Default::default()
            },
            EventChoice {
                result: 1, // 失败
                value: ActionValue {
                    status_pt: [5, 5, 5, 5, 5, 50],
                    vital: 30,
                    ..Default::default()
                },
                ..Default::default()
            },
        ];

        // 成功（rmj_results=true）→ result=2 分支
        let picked = select_rmj_choice_by_result(&choices, Some(true)).unwrap();
        println!("成功分支 result={}, value={:?}", picked.result, picked.value);
        assert_eq!(picked.result, 2);
        assert_eq!(picked.value.status_pt[5], 100);

        // 失败（rmj_results=false）→ result=1 分支
        let picked = select_rmj_choice_by_result(&choices, Some(false)).unwrap();
        println!("失败分支 result={}, value={:?}", picked.result, picked.value);
        assert_eq!(picked.result, 1);
        assert_eq!(picked.value.status_pt[5], 50);

        // 无结算结果 → 回退到第一个分支
        let picked = select_rmj_choice_by_result(&choices, None).unwrap();
        println!("无结果分支 result={}, value={:?}", picked.result, picked.value);
        assert_eq!(picked.result, 2);

        // 空 choices
        let picked = select_rmj_choice_by_result(&[], Some(true));
        assert!(picked.is_none());
        println!("空 choices 返回 None: {:?}", picked);
    }

    /// 验证 `rmj_event_year` 能正确返回年份索引
    #[test]
    fn test_rmj_event_year() {
        assert_eq!(rmj_event_year(401404), Some(0));
        assert_eq!(rmj_event_year(401405), Some(1));
        assert_eq!(rmj_event_year(401406), Some(2));
        assert_eq!(rmj_event_year(401407), None); // 育成结束事件不是 RMJ 事件
        assert_eq!(rmj_event_year(0), None);
        println!("rmj_event_year 映射验证通过");
    }

    /// 验证 RMJ 结算成功时，apply_event 选择 result=2 分支
    #[test]
    fn test_rmj_event_apply_success() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 把 vital 调到 0 避免上限截断干扰
        game.uma.vital = 0;
        // 设置 RMJ 成功状态
        game.ramen.rmj_results = vec![true];

        // 获取 401404 事件并 apply
        let event = find_rmj_event(0).expect("401404 事件应存在");
        let status_before = game.uma.five_status;
        let pt_before = game.uma.skill_pt;
        let vital_before = game.uma.vital;
        println!(
            "应用前: status={:?}, PT={}, vital={}",
            status_before, pt_before, vital_before
        );

        let mut rng = StdRng::seed_from_u64(42);
        game.apply_event(&event, 0, &mut rng)?;

        let status_after = game.uma.five_status;
        let pt_after = game.uma.skill_pt;
        let vital_after = game.uma.vital;
        println!(
            "应用后: status={:?}, PT={}, vital={}",
            status_after, pt_after, vital_after
        );

        // 成功分支应该：速+10, 耐+10, 力+10, 根+10, 智+10, pt+100, vital+33
        for i in 0..5 {
            assert_eq!(status_after[i] - status_before[i], 10, "属性 {i} 增量应为 10");
        }
        assert_eq!(pt_after - pt_before, 100);
        assert_eq!(vital_after - vital_before, 33);
        println!("RMJ 成功分支效果验证通过");

        Ok(())
    }

    /// 验证 RMJ 结算失败时，apply_event 选择 result=1 分支
    #[test]
    fn test_rmj_event_apply_fail() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 把 vital 调到 0 避免上限截断干扰
        game.uma.vital = 0;
        // 设置 RMJ 失败状态
        game.ramen.rmj_results = vec![false];

        let event = find_rmj_event(0).expect("401404 事件应存在");
        let status_before = game.uma.five_status;
        let pt_before = game.uma.skill_pt;
        let vital_before = game.uma.vital;
        println!(
            "RMJ 失败前: status={:?}, PT={}, vital={}",
            status_before, pt_before, vital_before
        );

        let mut rng = StdRng::seed_from_u64(42);
        game.apply_event(&event, 0, &mut rng)?;

        let status_after = game.uma.five_status;
        let pt_after = game.uma.skill_pt;
        let vital_after = game.uma.vital;
        println!(
            "RMJ 失败后: status={:?}, PT={}, vital={}",
            status_after, pt_after, vital_after
        );

        // 失败分支应该：速+5, 耐+5, 力+5, 根+5, 智+5, pt+50, vital+30
        for i in 0..5 {
            assert_eq!(status_after[i] - status_before[i], 5, "属性 {i} 增量应为 5");
        }
        assert_eq!(pt_after - pt_before, 50);
        assert_eq!(vital_after - vital_before, 30);
        println!("RMJ 失败分支效果验证通过");

        Ok(())
    }

    /// 验证 RMJ 结算后立即 apply 对应事件（在 turn=23 末触发，而非 turn=24 末）
    #[test]
    fn test_rmj_event_immediate_apply_at_turn_23() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 把 vital 调到 0 避免上限截断干扰
        game.uma.vital = 0;
        // 手动模拟 turn=23 RMJ 结算
        game.base.turn = 23;
        game.stage = RamenStage::NextTurn;

        // RMJ 结算前：unresolved 应该为空
        assert!(game.base.unresolved_events.is_empty());

        let pt_before = game.uma.skill_pt;
        let status_before = game.uma.five_status;

        // 触发 next() 中的 RMJ 结算逻辑
        // 注意：turn=23 的 RMJ 结算后会进入 RegionSelect 阶段（不是 advance_turn）
        game.next();
        println!("RMJ 结算后 turn={}, stage={:?}", game.base.turn, game.stage);

        // 验证 RMJ 已结算（rmj_results 写入）
        assert_eq!(game.ramen.rmj_results, vec![false], "默认 PT=0 < 1500 应失败");

        // turn=23 的 RMJ 结算后会进入 RegionSelect 阶段（地区选择是回合 23 末的特殊阶段）
        assert!(
            matches!(game.stage, RamenStage::RegionSelect),
            "RMJ 后应进入 RegionSelect 阶段（turn=23 末）"
        );

        // 验证 RMJ 失败分支已立即应用：pt 增加 50
        let pt_after = game.uma.skill_pt;
        println!("RMJ 结算前 PT={}, 结算后 PT={}", pt_before, pt_after);
        assert_eq!(pt_after - pt_before, 50, "RMJ 失败分支应加 50pt");

        // 验证 status[0] 增加 5（RMJ 失败分支）
        assert_eq!(game.uma.five_status[0] - status_before[0], 5);

        println!("RMJ 事件在 turn=23 末立即 apply 验证通过");

        Ok(())
    }

    /// 验证 RMJ 结算后 scenario_pt 归零，下一年重新累计
    #[test]
    fn test_scenario_pt_reset_after_rmj() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 模拟 turn=23 的 RMJ 结算：先设置 scenario_pt = 2500
        game.base.turn = 23;
        game.stage = RamenStage::NextTurn;
        game.ramen.scenario_pt = 2500;
        let pt_before = game.ramen.scenario_pt;
        println!("RMJ 结算前 scenario_pt = {}", pt_before);

        // 触发 next() 中的 RMJ 结算逻辑
        game.next();

        // 验证 scenario_pt 已归零
        assert_eq!(
            game.ramen.scenario_pt, 0,
            "RMJ 结算后 scenario_pt 应归零（实际 {}）",
            game.ramen.scenario_pt
        );
        println!("RMJ 结算后 scenario_pt = {}（归零成功）", game.ramen.scenario_pt);

        Ok(())
    }

    /// 吃面 PT 增量 / `eat_count += 1` 延后到 NextTurn 阶段（不立即生效）。
    ///
    /// 回归吃面前后 `scenario_pt` 的语义边界：
    /// - `ground_ramen_effects` 后：`scenario_pt` / `eat_count` **不变**（仅设 `current_ramen` /
    ///   消耗诀窍 / 分身 / 羁绊效果）
    /// - `calc_ramen_training_effect` 用"吃面前 PT"算 `ramen_pt_effect` 档位
    ///   （关键：避免本次吃面立即抬高档位）
    /// - `NextTurn` 阶段才累加 `scenario_pt += pt_gain`、`eat_count += 1`
    #[test]
    fn test_eat_ramen_pt_gain_defers_to_next_turn() -> Result<()> {
        use crate::gamedata::ramen::RAMENDATA;

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // 年 1 turn=5；scenario_pt = 900 落在 pt_min=500 档（xunlian=5），
        // 距 pt_min=1000 档（xunlian=8）差 100；吃面后 scenario_pt 若仍 = 900
        // 则档位不变；若错误地立即 +300 → PT=1200 → 档位会跳到 pt_min=1000 → xunlian+3。
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 5;
        game.stage = RamenStage::Train;
        game.ramen.scenario_pt = 900;
        game.ramen.eat_count = 0;
        // 手动给足诀窍（正常吃面回合由 init_feeling_stocks 在 turn 2/24/48 触发）；
        // TEST_DECK 含友人卡 303054 = 新友人(30305)，init_val=2/类型
        game.ramen.feeling_stock = [5, 5, 5];

        let pt_before = game.ramen.scenario_pt;
        let eat_before = game.ramen.eat_count;
        println!("吃面前: PT={} eat={}", pt_before, eat_before);

        // 先记录"吃面前"的拉面效果（current_ramen = None）作为基线
        let effect_before =
            crate::game::ramen::effects::calc_ramen_training_effect(&game, 0, false);
        let xunlian_baseline = effect_before.xunlian;
        println!("吃面前 xunlian 基线 = {} (仅 ramen_pt_effect 贡献)", xunlian_baseline);

        // 吃面：地区 0 + 不替换隐藏风味
        game.ramen.pending_ramen = Some(0);
        game.ramen.pending_special_targets = [0, 0, 0];

        let mut rng = StdRng::seed_from_u64(42);
        game.ground_ramen_effects(&mut rng)?;

        let mut c = Checks::new();
        println!(
            "吃面 ground 后: PT={} eat={} current_ramen={:?}",
            game.ramen.scenario_pt, game.ramen.eat_count, game.ramen.current_ramen
        );
        c.check(
            game.ramen.scenario_pt == pt_before,
            "ground_ramen_effects 不应立即增加 scenario_pt",
        );
        c.check(
            game.ramen.eat_count == eat_before,
            "ground_ramen_effects 不应立即 eat_count += 1",
        );
        c.check(
            game.ramen.current_ramen == Some(0),
            "ground_ramen_effects 应设置 current_ramen = Some(0)",
        );

        // 关键回归点：吃面后 calc_ramen_training_effect 的 ramen_pt_effect 档位
        // 必须用吃面前 PT（900，pt_min=500 档），xunlian 增量仅来自 basic + region。
        // 错误实现下 scenario_pt=1200 → pt_min=1000 档 → xunlian 比基线多 3（5→8）。
        let ramen_data = global!(RAMENDATA);
        let pt_tier_correct = ramen_data
            .ramen_pt_effect
            .iter()
            .rposition(|pe| pe.pt_min <= pt_before)
            .unwrap_or(0);
        let pt_tier_wrong = ramen_data
            .ramen_pt_effect
            .iter()
            .rposition(|pe| pe.pt_min <= pt_before + 300)
            .unwrap_or(0);
        let pt_xunlian_correct = ramen_data.ramen_pt_effect[pt_tier_correct].xunlian;
        let pt_xunlian_wrong = ramen_data.ramen_pt_effect[pt_tier_wrong].xunlian;
        println!(
            "ramen_pt_effect 档位: 正确 pt_min={} xunlian={} / 错误 pt_min={} xunlian={}",
            ramen_data.ramen_pt_effect[pt_tier_correct].pt_min,
            pt_xunlian_correct,
            ramen_data.ramen_pt_effect[pt_tier_wrong].pt_min,
            pt_xunlian_wrong,
        );
        // 吃面前后拉面效果增量的预期值：
        // 1) 正确：basic.xunlian + region_xunlian（ramen_pt_effect 档位不变 → 增量不含 3）
        // 2) 错误：basic.xunlian + region_xunlian + 3（PT 提前跳档 → 增量多 +3）
        let basic_year1 = &ramen_data.ramen_basic_effect[0];
        let region0 = &ramen_data.ramen_region_effect[0];
        let expected_delta_correct = basic_year1.xunlian + region0.xunlian;
        let expected_delta_wrong = basic_year1.xunlian + region0.xunlian + (pt_xunlian_wrong - pt_xunlian_correct);
        println!(
            "吃面后 xunlian 增量: 正确期望={} / 错误期望={} (差值 {})",
            expected_delta_correct,
            expected_delta_wrong,
            expected_delta_wrong - expected_delta_correct,
        );

        let effect_after =
            crate::game::ramen::effects::calc_ramen_training_effect(&game, 0, false);
        let delta = effect_after.xunlian - xunlian_baseline;
        println!(
            "calc_ramen_training_effect xunlian: 吃面前={} 吃面后={} 增量={}",
            xunlian_baseline, effect_after.xunlian, delta
        );
        c.check(
            delta == expected_delta_correct,
            &format!(
                "calc_ramen_training_effect 用吃面前 PT 算 ramen_pt_effect 档位 \
                 (期望增量 {} / 错误增量 {} = basic+region + 跳档 +{})",
                expected_delta_correct,
                expected_delta_wrong,
                pt_xunlian_wrong - pt_xunlian_correct,
            ),
        );

        // 手动触发 NextTurn 阶段，验证 PT 增量 / eat_count += 1 在此处生效
        game.stage = RamenStage::NextTurn;
        game.next();
        let pt_after_expected = pt_before + 300; // 年 1 第一面 gain_pt_base=300, eat=0 → 300
        println!(
            "NextTurn 后: PT={} eat={}",
            game.ramen.scenario_pt, game.ramen.eat_count
        );
        c.check(
            game.ramen.scenario_pt == pt_after_expected,
            "NextTurn 后 scenario_pt 应 += 300（年 1 第一面）",
        );
        c.check(
            game.ramen.eat_count == 1,
            "NextTurn 后 eat_count 应 += 1",
        );
        c.finish()
    }

    /// RMJ 清零前必须把当年 PT / 吃面次数写入 `yearly_*`。
    #[test]
    fn test_rmj_archives_yearly_counters_before_reset() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut c = Checks::new();
        for (turn, year_idx) in [(23, 0usize), (47, 1), (71, 2)] {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.base.turn = turn;
            game.stage = RamenStage::NextTurn;
            game.ramen.scenario_pt = 1234;
            game.ramen.eat_count = 7;
            println!("turn={turn} 归档前 live PT={} eat={}", game.ramen.scenario_pt, game.ramen.eat_count);
            game.next();
            println!(
                "turn={turn} 归档后 yearly_pt={:?} yearly_eat={:?} live PT={} eat={}",
                game.ramen.yearly_scenario_pt,
                game.ramen.yearly_eat_count,
                game.ramen.scenario_pt,
                game.ramen.eat_count
            );
            c.check(
                game.ramen.yearly_scenario_pt[year_idx] == 1234,
                &format!("turn {turn} yearly_scenario_pt[{year_idx}] == 1234")
            );
            c.check(
                game.ramen.yearly_eat_count[year_idx] == 7,
                &format!("turn {turn} yearly_eat_count[{year_idx}] == 7")
            );
            c.check(game.ramen.scenario_pt == 0, &format!("turn {turn} live scenario_pt 归零"));
            c.check(game.ramen.eat_count == 0, &format!("turn {turn} live eat_count 归零"));
            for other in 0..3 {
                if other == year_idx {
                    continue;
                }
                c.check(
                    game.ramen.yearly_scenario_pt[other] == 0,
                    &format!("turn {turn} 其它年 {other} 的 PT 仍为 0")
                );
                c.check(
                    game.ramen.yearly_eat_count[other] == 0,
                    &format!("turn {turn} 其它年 {other} 的 eat_count 仍为 0")
                );
            }
        }
        c.finish()
    }

    /// 验证 generate_events 在 turn=0 时返回 400000400 马娘登场事件
    #[test]
    fn test_generate_events_uma_debut() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(42);
        // turn=0 应触发马娘登场
        game.base.turn = 0;
        let events = game.generate_events(&mut rng);
        println!(
            "turn=0 事件数: {}, IDs: {:?}",
            events.len(),
            events.iter().map(|e| e.id).collect::<Vec<_>>()
        );
        assert!(!events.is_empty(), "turn=0 应有事件");
        assert_eq!(events[0].id, 400000400, "turn=0 第一个事件应是马娘登场");

        Ok(())
    }

    /// 验证 generate_events 在 turn=24 时返回 4009 经典年新年事件
    #[test]
    fn test_generate_events_classic_newyear() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(42);
        game.base.turn = 24;
        let events = game.generate_events(&mut rng);
        println!(
            "turn=24 事件数: {}, IDs: {:?}",
            events.len(),
            events.iter().map(|e| e.id).collect::<Vec<_>>()
        );
        assert!(!events.is_empty(), "turn=24 应有事件");
        assert_eq!(events[0].id, 4009, "turn=24 第一个事件应是经典年新年");

        Ok(())
    }

    /// 验证 generate_events 在 turn=48 时返回 4010 古马年新年事件
    #[test]
    fn test_generate_events_ancient_newyear() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(42);
        game.base.turn = 48;
        let events = game.generate_events(&mut rng);
        println!(
            "turn=48 事件数: {}, IDs: {:?}",
            events.len(),
            events.iter().map(|e| e.id).collect::<Vec<_>>()
        );
        assert!(!events.is_empty(), "turn=48 应有事件");
        assert_eq!(events[0].id, 4010, "turn=48 第一个事件应是古马年新年");

        Ok(())
    }

    /// 验证 add_mandatory_events 在 turn=48 时将 ticket(4011) push 到 unresolved_events
    #[test]
    fn test_add_mandatory_events_ticket_at_48() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 48;
        let mut events = vec![];
        game.add_mandatory_events(&mut events)?;
        // turn=48 没有友人解锁就没有友人事件
        println!(
            "turn=48 同步事件数: {}, unresolved 数: {}",
            events.len(),
            game.base.unresolved_events.len()
        );
        // 4011 (ticket) 应在 unresolved_events 中
        assert!(game.base.unresolved_events.iter().any(|e| e.id == 4011));
        println!("turn=48 ticket(4011) 已在 unresolved_events 中");

        Ok(())
    }

    /// 验证 add_mandatory_events 在 turn=77 时将 ending(5011) 和 401407 push 到 unresolved_events
    #[test]
    fn test_add_mandatory_events_ending_at_77() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 77;
        let mut events = vec![];
        game.add_mandatory_events(&mut events)?;
        println!(
            "turn=77 同步事件数: {}, unresolved 数: {}",
            events.len(),
            game.base.unresolved_events.len()
        );

        // ending(5011) 和 401407 应在 unresolved_events 中
        let unresolved_ids: Vec<u32> = game.base.unresolved_events.iter().map(|e| e.id).collect();
        println!("turn=77 unresolved_events IDs: {:?}", unresolved_ids);
        assert!(unresolved_ids.contains(&5011), "5011 应在 unresolved_events");
        assert!(unresolved_ids.contains(&401407), "401407 应在 unresolved_events");

        Ok(())
    }

    /// 验证超级拉面回合（turn=72-77）的 vital/motivation 每回合自动恢复
    /// + saihou（赛后加成）仅 turn=72 一次性 +100（之后回合不重复累加）
    #[test]
    fn test_super_ramen_base_effect_vital_motivation() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 跳到 URA 第一个回合（turn=72）
        game.base.turn = 72;
        game.add_friend_and_npcs()?;
        // 设置 super_ramen 选项（必要条件之一）
        game.ramen.super_ramen = Some(1);

        // 清零关键字段以便观察增量
        game.uma.vital = 50;
        game.uma.motivation = 2;
        let race_bonus_before = game.uma.race_bonus;
        let vital_before = game.uma.vital;
        let motivation_before = game.uma.motivation;

        // 调用 run_begin（vital/motivation + race_bonus 一次性+100）
        let trainer = RandomTrainer;
        let mut rng = StdRng::from_os_rng();
        game.run_begin(&trainer, &mut rng)?;

        let race_bonus_after_run_begin = game.uma.race_bonus;
        let vital_after = game.uma.vital;
        let motivation_after = game.uma.motivation;
        println!(
            "超级拉面前: vital={}, motivation={}, race_bonus={}",
            vital_before, motivation_before, race_bonus_before
        );
        println!(
            "超级拉面 run_begin 后: vital={}, motivation={}, race_bonus={}",
            vital_after, motivation_after, race_bonus_after_run_begin
        );

        // 验证 turn=72：vital+20, motivation+1, race_bonus+100（一次性）
        assert_eq!(vital_after - vital_before, 20, "vital 应 +20");
        assert_eq!(motivation_after - motivation_before, 1, "motivation 应 +1");
        assert_eq!(
            race_bonus_after_run_begin - race_bonus_before,
            100,
            "turn=72 race_bonus 应一次性 +100"
        );

        println!("超级拉面 turn=72 一次性恢复 + vital/motivation 每回合恢复验证通过");

        Ok(())
    }

    /// 验证 saihou 仅在 turn=72 一次性 +100，turn=73-77 不再累加
    ///
    /// 模拟 turn=72-75 连续运行，观察 race_bonus 只在 turn=72 +100，后续回合不变。
    #[test]
    fn test_super_ramen_saihou_one_time_only() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.ramen.super_ramen = Some(1);

        let race_bonus_initial = game.uma.race_bonus;
        println!("初始 race_bonus: {}", race_bonus_initial);

        let trainer = RandomTrainer;
        // 模拟连续多个 URA 回合（turn=72-75），观察 race_bonus 增量
        for turn in 72..=75 {
            game.base.turn = turn;
            // 重新设置 vital/motivation 以避免上限截断干扰
            game.uma.vital = 50;
            game.uma.motivation = 2;

            let race_bonus_before = game.uma.race_bonus;
            let mut rng = StdRng::from_os_rng();
            game.run_begin(&trainer, &mut rng)?;
            let race_bonus_after = game.uma.race_bonus;
            let expected_increment = if turn == 72 { 100 } else { 0 };
            println!(
                "turn={} 前 race_bonus={}, 后 race_bonus={}, 期望增量={}",
                turn, race_bonus_before, race_bonus_after, expected_increment
            );
            assert_eq!(
                race_bonus_after - race_bonus_before,
                expected_increment,
                "turn={} race_bonus 增量应={}",
                turn,
                expected_increment
            );
        }

        // 最终 race_bonus 应为 initial + 100（仅 turn=72 加了一次）
        assert_eq!(
            game.uma.race_bonus,
            race_bonus_initial + 100,
            "连续 4 回合 URA 后 race_bonus 仅 +100"
        );

        println!("saihou 一次性 +100（不跨回合累积）验证通过");

        Ok(())
    }

    /// `apply_super_ramen_saihou` 幂等：条件满足时合计只 +saihou 一次
    ///
    /// 覆盖「模拟 run_begin」与「协议重放 into_game」两条路径合计不重复加：
    /// - turn<72 或未选超级拉面（None）→ 不生效；
    /// - turn>=72 且已选超级拉面 → 首次 +100，二次调用不再加。
    #[test]
    fn test_apply_super_ramen_saihou_idempotent() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.ramen.super_ramen = Some(1);

        // turn<72：不生效
        game.base.turn = 71;
        let base_bonus = game.uma.race_bonus;
        assert!(!game.apply_super_ramen_saihou(), "turn<72 不应生效");
        assert_eq!(game.uma.race_bonus, base_bonus, "turn<72 race_bonus 不应变");

        // turn=72：首次生效 +100
        game.base.turn = 72;
        assert!(game.apply_super_ramen_saihou(), "turn>=72 且已选超级拉面应首次生效");
        let after_first = game.uma.race_bonus;
        assert_eq!(after_first - base_bonus, 100, "应一次性 +100");

        // 二次调用（模拟重放已生效后 run_begin 仍被调）：不重复加
        assert!(!game.apply_super_ramen_saihou(), "二次调用不应生效");
        assert_eq!(game.uma.race_bonus, after_first, "race_bonus 不应再涨");

        // 未选超级拉面（None）：不生效
        let mut game2 = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game2.base.turn = 75;
        assert!(!game2.apply_super_ramen_saihou(), "未选超级拉面不应生效");

        println!("apply_super_ramen_saihou 幂等性验证通过");
        Ok(())
    }

    // ========== hint_special 单元测试 ==========

    /// 创建一个 hint_special 相关测试用的 RamenGame
    ///
    /// 关键设置：deck_can_split=true（支援卡种类>=4），年份=3（hint_special=true）。
    fn make_hint_special_test_game() -> RamenGame {
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT).expect("newgame 失败");
        // 设置为第三年且确保支援卡种类>=4
        game.base.turn = 60; // year 3
        game.deck_can_split = true;
        game
    }

    /// 不吃面时 hint_special 不应生效
    #[test]
    fn test_hint_special_inactive_without_ramen() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let game = make_hint_special_test_game();
        assert!(!game.calc_hint_special_active(game.ramen.current_ramen), "不吃面时 hint_special 必须为 false");
        // 任何 train 都应返回 false
        for train in 0..5 {
            assert!(
                !game.is_hint_special_active_for_train(train),
                "不吃面时 train={} 的 hint_special 必须为 false",
                train
            );
        }
        println!("不吃面时 hint_special 不生效 ✓");
        Ok(())
    }

    /// 吃面但不是第3年时 hint_special 不应生效
    #[test]
    fn test_hint_special_inactive_year1_2() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = make_hint_special_test_game();
        // year 1
        game.base.turn = 5;
        game.ramen.current_ramen = Some(5);
        assert!(
            !game.calc_hint_special_active(game.ramen.current_ramen),
            "year1 吃面时 hint_special 必须为 false（basic.year0.hint_special=false）"
        );

        // year 2
        game.base.turn = 30;
        assert!(
            !game.calc_hint_special_active(game.ramen.current_ramen),
            "year2 吃面时 hint_special 必须为 false（basic.year1.hint_special=false）"
        );

        println!("year1/year2 吃面时 hint_special 不生效 ✓");
        Ok(())
    }

    /// 第3年吃面时 hint_special 应生效
    #[test]
    fn test_hint_special_active_year3() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = make_hint_special_test_game();
        game.base.turn = 60;
        game.ramen.current_ramen = Some(5);
        assert!(
            game.calc_hint_special_active(game.ramen.current_ramen),
            "year3 + 吃面 + 支援卡种类>=4 时 hint_special 应生效"
        );

        // 检查 at_trains 是否正确（region 5 的 at_trains）
        let at_trains = game.calc_hint_special_at_trains(game.ramen.current_ramen);
        println!("region 5 at_trains={:?}", at_trains);
        // ramen_region_effect[5] 的 at_trains=[0,1,2,3,4]（全位置）
        assert_eq!(at_trains, vec![0, 1, 2, 3, 4]);

        // 所有 train 都应激活 hint_special
        for train in 0..5 {
            assert!(
                game.is_hint_special_active_for_train(train),
                "全位置面时 train={} 应激活 hint_special",
                train
            );
        }
        println!("year3 + 全位置面 + 支援卡种类>=4 时 hint_special 对所有 train 生效 ✓");
        Ok(())
    }

    /// hint_special 只在 at_trains 中的 train 生效
    #[test]
    fn test_hint_special_only_at_listed_trains() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = make_hint_special_test_game();
        game.base.turn = 60;
        // region 0 的 at_trains=[0]，只对速训练生效
        game.ramen.current_ramen = Some(0);
        assert!(game.calc_hint_special_active(game.ramen.current_ramen), "hint_special 应生效");

        assert!(
            game.is_hint_special_active_for_train(0),
            "train=0 在 at_trains=[0] 中应激活"
        );
        for train in 1..5 {
            assert!(
                !game.is_hint_special_active_for_train(train),
                "train={} 不在 at_trains=[0] 中应不激活",
                train
            );
        }

        let at_trains = game.calc_hint_special_at_trains(game.ramen.current_ramen);
        println!("region 0 at_trains={:?}", at_trains);
        let mut checks = Checks::default();
        checks.check(at_trains == [0], "region 0 只包含速训练位置");
        game.ramen.current_ramen = None;
        checks.check(game.calc_hint_special_at_trains(game.ramen.current_ramen).is_empty(), "未吃面时训练位置集合为空");
        for ramen in [None, Some(0), Some(5)] {
            let mut reference = game.clone();
            reference.ramen.current_ramen = ramen;
            for train in 0..5 {
                checks.check(
                    game.is_hint_special_active_for_train_with_ramen(train, ramen)
                        == reference.is_hint_special_active_for_train(train),
                    "借用基础局面的候选 Hint 与真实吃面状态一致"
                );
            }
        }
        game.ramen.current_ramen = Some(global!(RAMENDATA).ramen_region_effect.len());
        checks.check(game.calc_hint_special_at_trains(game.ramen.current_ramen).is_empty(), "地区索引不存在时训练位置集合为空");
        println!("hint_special 仅在 at_trains 训练位置生效 ✓");
        checks.finish()
    }

    /// 支援卡种类 < 4 时 hint_special 不应生效
    #[test]
    fn test_hint_special_inactive_low_card_types() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = make_hint_special_test_game();
        game.base.turn = 60;
        game.ramen.current_ramen = Some(5);
        // 模拟支援卡种类 < 4（只有3种）
        game.card_type_count = [1, 1, 1, 0, 0, 0, 0];
        game.deck_can_split = false;
        assert!(
            !game.calc_hint_special_active(game.ramen.current_ramen),
            "支援卡种类<4 时 hint_special 必须为 false"
        );
        println!("支援卡种类<4 时 hint_special 不生效 ✓");
        Ok(())
    }

    // ========== ManualTrainer 完整游戏测试 ==========

    /// 使用 ManualTrainer 完成完整游戏的测试
    ///
    /// `ManualTrainer` 真实模式依赖 `inquire` 终端交互，不适合自动化测试。
    /// 本测试使用 `ManualTrainer::with_mock_inputs(vec![])`（空队列 + PickFirst fallback）：
    /// - mock 队列为空，所有决策自动选第一个候选
    /// - 验证拉面杯从开局到育成的完整流程能跑通
    /// - 这相当于"模拟一个总是选第一个候选的玩家"
    #[test]
    fn test_manual_trainer_full_game() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error"); // 静默
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 空 mock 队列：所有决策走 PickFirst fallback（选第一个候选）
        let trainer = ManualTrainer::with_mock_inputs(vec![]);
        let mut rng = StdRng::seed_from_u64(20240816);

        println!("=== ManualTrainer 完整游戏测试 ===");
        println!("卡组: {:?}", TEST_DECK);
        println!("种子: 20240816");

        // 测试场景下不再 disable_log：cargo test 已隔离
        game.run_full_game(&trainer, &mut rng)?;

        // 验证游戏确实跑完了（最终回合应 == max_turn）
        let max_turn = game.max_turn();
        println!("\n=== 育成结果 ===");
        println!("最终回合: {} (max_turn={})", game.turn(), max_turn);
        assert_eq!(game.turn(), max_turn, "应跑完所有回合");

        // 验证拉面杯特有状态
        println!("剧本PT: {}", game.ramen.scenario_pt);
        println!("RMJ结果: {:?}", game.ramen.rmj_results);
        println!("地区选择: {:?}", game.ramen.selected_regions);
        println!("超级拉面选择: {:?}", game.ramen.super_ramen);
        println!(
            "诀窍库存: A={} B={} C={}",
            game.ramen.feeling_stock[0], game.ramen.feeling_stock[1], game.ramen.feeling_stock[2]
        );
        println!("隐藏风味: {}", game.ramen.special_feeling);
        let score = game.uma.calc_score();
        println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);

        // 验证基础状态合理性
        assert!(game.uma.vital >= 0, "体力应非负: {}", game.uma.vital);
        assert!(score >= 0, "评分应非负: {score}");

        println!("ManualTrainer 完整流程跑通 ✓");
        Ok(())
    }

    /// 使用 ManualTrainer 测试 hint_special 路径（第3年 + 吃面 + 支援卡种类>=4）
    ///
    /// 主要验证 game 流程不会因为全员 hint 而 panic 或 deadlock。
    #[test]
    fn test_manual_trainer_hint_special_path() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let trainer = ManualTrainer::with_mock_inputs(vec![]);
        let mut rng = StdRng::seed_from_u64(20240817);

        // 跳到第3年回合开始
        game.add_friend_and_npcs()?;
        game.add_reporter();
        game.base.turn = 60; // year 3
        game.deck_can_split = true;

        // 测试场景下不再 disable_log：cargo test 已隔离
        // 跑几个回合观察 hint_special 流程
        let mut turn_count = 0;
        loop {
            let max_turn = game.max_turn();
            if game.turn() >= max_turn {
                break;
            }
            turn_count += 1;
            if turn_count > 5 {
                // 限制回合数避免测试太长
                break;
            }
            game.run_full_game(&trainer, &mut rng)?;
            if game.turn() >= max_turn {
                break;
            }
        }

        println!("第3年跑完 {} 轮无 panic", turn_count);
        println!(
            "最终回合: {}, is_hint_special_active={}",
            game.turn(),
            game.calc_hint_special_active(game.ramen.current_ramen)
        );
        println!("ManualTrainer + hint_special 路径未崩溃 ✓");
        Ok(())
    }

    /// 第 1 年地区选择走真实阶段路径，从全枚举候选里落地
    ///
    /// 本测试**不再**尝试设置 `ramen_region_strategy=Fixed`：
    /// `init_global_with_config` 幂等，globals 已初始化时会直接返回 `Ok(())` 并
    /// 丢弃传入 config，测试并行跑在同一进程里根本设不进去——旧版本因此一直在
    /// 对着默认配置空转，声称验证了 Fixed 却什么都没验。
    /// 「Fixed 仅第 3 年生效」现由
    /// `test_year3_fixed_list_actions_single_candidate` 用纯函数显式传参覆盖。
    #[test]
    fn test_year1_region_select_uses_full_enumeration() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.base.turn = 2;
        game.stage = RamenStage::RegionSelect;
        let mut rng = StdRng::seed_from_u64(20260819);
        let trainer = ManualTrainer::with_mock_inputs(vec![]);

        let candidates = game.list_actions()?;
        println!("第 1 年 RegionSelect 候选数={}", candidates.len());
        let mut c = Checks::new();
        c.check(candidates.len() == 10, "第 1 年枚举 C(5,3)=10 个组合");

        game.run_region_select(&trainer, &mut rng, 0)?;
        println!("落地地区={:?}", game.ramen.selected_regions);
        c.check(
            game.ramen.selected_regions == [0, 1, 2],
            "ManualTrainer 无输入时取候选 0，即组合 [0,1,2]"
        );
        c.finish()
    }

    /// 回合 0-1 / 超级拉面回合应跳过 RamenSelect/SpecialSelect，直接从 Distribute 跳到 Train
    ///
    /// 短路规则：
    /// - turn < 2：剧本机制未启用，无法吃面
    /// - turn ∈ [72, 77]：超级拉面自动生效
    /// 其他回合仍走 Distribute → RamenSelect → SpecialSelect → Train
    #[test]
    fn test_skip_ramen_select_for_turn_0_1_and_super_ramen() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // 验证 1：回合 0（剧本机制未启用）应跳过 RamenSelect
        {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.add_friend_and_npcs()?;
            game.base.turn = 0;
            game.stage = RamenStage::Distribute;
            Game::next(&mut game);
            assert_eq!(
                game.stage,
                RamenStage::Train,
                "回合 0 应从 Distribute 直接跳到 Train（跳过 RamenSelect）"
            );
        }

        // 验证 2：回合 1（仍剧本机制未启用）应跳过 RamenSelect
        {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.add_friend_and_npcs()?;
            game.base.turn = 1;
            game.stage = RamenStage::Distribute;
            Game::next(&mut game);
            assert_eq!(
                game.stage,
                RamenStage::Train,
                "回合 1 应从 Distribute 直接跳到 Train（跳过 RamenSelect）"
            );
        }

        // 验证 3：回合 2（剧本机制启用）应正常走 RamenSelect
        {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.add_friend_and_npcs()?;
            game.base.turn = 2;
            game.stage = RamenStage::Distribute;
            Game::next(&mut game);
            assert_eq!(
                game.stage,
                RamenStage::RamenSelect,
                "回合 2 应正常从 Distribute 走到 RamenSelect"
            );
        }

        // 验证 4：超级拉面回合(72-77)应跳过 RamenSelect
        for turn in [72, 73, 74, 75, 76, 77] {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.add_friend_and_npcs()?;
            game.base.turn = turn;
            game.stage = RamenStage::Distribute;
            Game::next(&mut game);
            assert_eq!(
                game.stage,
                RamenStage::Train,
                "回合 {} 应从 Distribute 直接跳到 Train（超级拉面自动生效）",
                turn
            );
        }

        // 验证 5：回合 71（仍正常吃面）应正常走 RamenSelect
        {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.add_friend_and_npcs()?;
            game.base.turn = 71;
            game.stage = RamenStage::Distribute;
            Game::next(&mut game);
            assert_eq!(
                game.stage,
                RamenStage::RamenSelect,
                "回合 71 应正常从 Distribute 走到 RamenSelect（超级拉面尚未生效）"
            );
        }

        println!("回合 0/1/72-77 短路规则全部通过 ✓");
        Ok(())
    }

    /// 回归：人头下标与卡组槽位的映射（羁绊双份一致性）
    ///
    /// 拉面的 `init_persons` 过滤掉友人卡，导致 `persons[5]` = 理事长、`persons[6]` = 友人卡，
    /// 而 `deck[5]` 是友人卡。修复前 `add_friendship` 的 `person_index < 6` 守卫会把**理事长**
    /// 的羁绊写进 `deck[5].friendship`（友人卡那份拷贝），后者又被
    /// `SupportCard::calc_training_effect` 用于固有解锁判定。
    ///
    /// 修复后 `add_friendship` 按 `card_id` 反查卡组槽位，本测试跑完整一局校验：
    /// 友人卡的两份羁绊一致，前 5 张训练卡的两份羁绊一致。
    #[test]
    fn test_person_deck_index_mapping_full_game() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("warn");
        let _ = init_global();

        // 开局快照
        let fresh = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let friend_card = &fresh.deck[5];
        println!(
            "deck[5] = {} 初始羁绊={} unique_type={} unique_param={:?}",
            friend_card.data.short_name(),
            friend_card.friendship,
            friend_card.data.unique_effect_type,
            friend_card.data.unique_effect_param
        );
        println!("开局 persons 布局:");
        for (i, p) in fresh.persons.iter().enumerate() {
            println!(
                "  persons[{i}] = {} type={:?} card_id={:?}",
                p.short_name(),
                p.person_type,
                p.card_id
            );
        }

        // 跑完整一局
        for run_idx in 0..3u64 {
            let (mut decision_rng, rule_master) = crate::bench::seeded_rngs(20260823, run_idx);
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.set_rule_master(rule_master);
            let trainer = crate::trainer::LoggingTrainer::new(
                crate::trainer::RamenHandwrittenTrainer::new(),
                rule_master
            );
            game.run_full_game(&trainer, &mut decision_rng)?;

            println!("--- run {run_idx} (seed={rule_master}) ---");
            println!("  persons[5] 理事长 羁绊 = {}", game.persons[5].friendship);
            println!(
                "  persons[6] {} 羁绊 = {}",
                game.persons[6].short_name(),
                game.persons[6].friendship
            );
            println!(
                "  deck[5] {} 羁绊 = {} (is_locked={})",
                game.deck[5].data.short_name(),
                game.deck[5].friendship,
                game.deck[5].is_locked
            );
            println!(
                "  一致性: deck[5]==persons[5]? {} / deck[5]==persons[6]? {}",
                game.deck[5].friendship == game.persons[5].friendship,
                game.deck[5].friendship == game.persons[6].friendship
            );
            for i in 0..5 {
                println!(
                    "  对照 idx{i}: deck={} persons={}",
                    game.deck[i].friendship, game.persons[i].friendship
                );
            }

            // 回归校验：友人卡的两份羁绊必须同步（修复前 deck[5] 跟的是理事长）
            println!(
                "  [{}] deck[5](友人卡) 的羁绊应与 persons[6](友人卡本人) 同步",
                check(game.deck[5].friendship == game.persons[6].friendship)
            );
            // 前 5 张训练卡本来就同序，作为对照必须始终一致
            let card_sync = (0..5).all(|i| game.deck[i].friendship == game.persons[i].friendship);
            println!("  [{}] 前 5 张训练卡的两份羁绊应一致", check(card_sync));
        }
        Ok(())
    }

    /// 回归：`default_calc_training_buff` 按 `card_id` 反查卡组，不再拿人头下标当卡组下标
    ///
    /// 修复前 `traits.rs` 用 `index < 6` 把人头下标当卡组下标。拉面里 `persons[5]` 是理事长、
    /// `persons[6]` 是友人卡，于是理事长会顶着友人卡的训练加成进训练，而友人卡本人被整个跳过。
    ///
    /// 修复后：理事长（无卡人头）的 buff 恒为空，友人卡拿到自己在 `deck` 里那张卡的效果。
    #[test]
    fn test_training_buff_person_deck_mapping() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("warn");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        println!(
            "persons[5]={} persons[6]={}",
            game.persons[5].short_name(),
            game.persons[6].short_name()
        );

        // 只放理事长（人头 5）
        game.distribution = vec![vec![]; 5];
        game.distribution[0] = vec![5];
        let buff_yayoi = game.calc_training_buff(0)?;
        println!("只放理事长(人头5) 的训练 buff: {:?}", buff_yayoi);

        // 只放友人卡（人头 6）
        game.distribution[0] = vec![6];
        let buff_friend = game.calc_training_buff(0)?;
        println!("只放友人卡(人头6) 的训练 buff: {:?}", buff_friend);

        // 对照：只放 1 号训练卡
        game.distribution[0] = vec![1];
        let buff_card = game.calc_training_buff(0)?;
        println!("只放 [智]青春(人头1) 的训练 buff: {:?}", buff_card);

        // 回归校验 1：理事长是无卡人头，反查不到卡组槽位，buff 必须为空
        println!(
            "[{}] 理事长不应携带任何支援卡加成",
            check(buff_yayoi == CardTrainingEffect::default())
        );

        // 回归校验 2：友人卡的 buff 必须等于它自己那张卡（deck[5]）算出的效果
        // 注意：走一遍 `CardTrainingEffect::add` 才能和聚合结果对齐（deyilv 不参与聚合）
        let mut friend_effect = game.deck[5].calc_training_effect(&game, 0);
        if !game.is_shining_at(6, 0) {
            friend_effect.youqing = 0.0;
        }
        let expect_friend = CardTrainingEffect::default().add(&friend_effect);
        println!("[{}] 友人卡(人头6) 的 buff 应来自 deck[5]", check(buff_friend == expect_friend));

        // 回归校验 3：训练卡对照组不受影响
        let mut card1_effect = game.deck[1].calc_training_effect(&game, 0);
        if !game.is_shining_at(1, 0) {
            card1_effect.youqing = 0.0;
        }
        let expect_card1 = CardTrainingEffect::default().add(&card1_effect);
        println!("[{}] 训练卡(人头1) 的 buff 应来自 deck[1]", check(buff_card == expect_card1));

        // 固有阈值判定读的是 deck[..].friendship。修复后理事长不再触发任何卡的固有。
        game.deck[5].friendship = 60;
        game.distribution[0] = vec![5];
        let buff_unlocked = game.calc_training_buff(0)?;
        println!("把 deck[5].friendship 抬到 60 后，理事长(人头5) 的训练 buff: {:?}", buff_unlocked);
        println!(
            "[{}] 理事长不应因 deck[5] 羁绊而解锁固有",
            check(buff_unlocked == CardTrainingEffect::default())
        );
        Ok(())
    }

    /// 回归：训练人数加成按 `PersonType` 计数，不再硬编码人头下标 6/7
    ///
    /// 旧实现 `filter(|p| **p != 6 && **p != 7)` 是温泉布局（卡 0-5、理事长 6、
    /// 记者 7）的下标常量。拉面的理事长在 5、友人卡在 6、NPC 从 7 起、记者在 12，
    /// 四项全部判反：理事长与记者被计入，友人卡与第一个 NPC 被排除。
    ///
    /// 本用例直接测 `count_training_persons`，不经过训练值公式——训练基础值
    /// 只有 11~15，`floor(x × 1.05)` 会把「差一个人」的效果整个吃掉，
    /// 拿 `ActionValue` 当判据测不出单人差异。
    #[test]
    fn test_count_training_persons_by_type() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.add_reporter();

        // 按类型定位，不写死下标——下标只作为本卡组的前提打印出来
        let find = |ty: PersonType| -> Result<i32> {
            game.persons
                .iter()
                .position(|p| p.person_type == ty)
                .map(|i| i as i32)
                .ok_or_else(|| anyhow!("找不到人头类型 {ty:?}"))
        };
        let card = find(PersonType::Card)?;
        let friend = find(PersonType::ScenarioCard)?;
        let npc = find(PersonType::Npc)?;
        let yayoi = find(PersonType::Yayoi)?;
        let reporter = find(PersonType::Reporter)?;
        println!("人头布局: 卡={card} 友人={friend} NPC={npc} 理事长={yayoi} 记者={reporter}");

        // 旧实现的判据，仅用于对照打印
        let legacy = |dist: &[i32]| dist.iter().filter(|p| **p != 6 && **p != 7).count();

        // (分布, 期望人数, 说明)
        let cases: Vec<(Vec<i32>, usize, &str)> = vec![
            (vec![], 0, "空分布"),
            (vec![-1, -1], 0, "「不出现」哨兵不计数"),
            (vec![yayoi], 0, "理事长不吃人数加成"),
            (vec![reporter], 0, "记者不吃人数加成"),
            (vec![friend], 1, "友人卡计入（旧实现漏计）"),
            (vec![npc], 1, "第一个 NPC 计入（旧实现漏计）"),
            (vec![card, card], 2, "分身按占位重复计数，不去重"),
            (vec![friend, friend], 2, "友人卡分身同样重复计数"),
            (vec![card, yayoi, friend, npc, reporter], 3, "混合：只有卡/友人/NPC 计入"),
            (vec![999], 0, "越界下标不计数也不 panic"),
        ];

        let mut c = Checks::new();
        for (dist, want, what) in cases {
            game.base.distribution = vec![dist.clone(), vec![], vec![], vec![], vec![]];
            let got = game.count_training_persons(0);
            println!("  {dist:?} -> 新={got} 期望={want} 旧={}", legacy(&dist));
            c.check(got == want, what);
        }
        c.finish()
    }

    /// 回归：人数计数改按类型判定后，温泉与 base 的结果逐位不变
    ///
    /// 这是本次改动能安全落到共享 `Game` trait 的前提：温泉人头恒为
    /// 卡 0-5、理事长 6、记者 7，`BasicGame` 更是只有卡 0-5 + 理事长 6
    /// （没有记者，旧实现里的 `!= 7` 对它一直是空操作）。
    /// 于是「排除下标 6/7」与「排除理事长/记者」在两者上是同一个集合。
    #[test]
    fn test_count_training_persons_onsen_unchanged() -> Result<()> {
        use crate::game::onsen::game::OnsenGame;

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = OnsenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut c = Checks::new();

        // 前提：下标 6/7 当且仅当是理事长/记者
        println!("温泉人头数 = {}", game.persons.len());
        c.check(game.persons.len() == 8, "温泉应为 6 张卡 + 理事长 + 记者");
        for (i, p) in game.persons.iter().enumerate() {
            let is_excluded_type = matches!(p.person_type, PersonType::Yayoi | PersonType::Reporter);
            println!("  人头 {i}: {:?}", p.person_type);
            c.check(is_excluded_type == (i == 6 || i == 7), "下标 6/7 当且仅当理事长/记者");
        }

        // 逐位比对新旧两种判据（不含负数：生产路径从不把 -1 写进 distribution）
        let samples: Vec<Vec<i32>> = vec![
            vec![],
            vec![0],
            vec![5],
            vec![6],
            vec![7],
            vec![0, 6],
            vec![0, 7],
            vec![6, 7],
            vec![0, 0],
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 6, 7],
        ];
        for dist in &samples {
            let legacy = dist.iter().filter(|p| **p != 6 && **p != 7).count();
            game.distribution = vec![dist.clone(), vec![], vec![], vec![], vec![]];
            let got = game.count_training_persons(0);
            println!("  {dist:?} -> 新={got} 旧={legacy}");
            c.check(got == legacy, "温泉上新旧计数必须相同");
        }

        // base：只有 6 张卡 + 理事长(6)，没有记者，旧判据里的 `!= 7` 一直是空操作
        let mut basic = crate::game::base::basic::BasicGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        println!("base 人头数 = {}", basic.persons.len());
        c.check(basic.persons.len() == 7, "base 应为 6 张卡 + 理事长，且没有记者");
        c.check(
            basic.persons.iter().all(|p| p.person_type != PersonType::Reporter),
            "base 不存在记者人头"
        );
        for dist in &samples {
            // base 只有 0..=6，下标 7 取不到，样本里含 7 的部分等价于被跳过
            let dist: Vec<i32> = dist.iter().copied().filter(|&p| p < 7).collect();
            let legacy = dist.iter().filter(|p| **p != 6 && **p != 7).count();
            basic.distribution = vec![dist.clone(), vec![], vec![], vec![], vec![]];
            let got = basic.count_training_persons(0);
            println!("  base {dist:?} -> 新={got} 旧={legacy}");
            c.check(got == legacy, "base 上新旧计数必须相同");
        }
        c.finish()
    }
    /// 回归：地区拉面分身是 per-训练位语义，且满员是规格内跳过
    ///
    /// 与超级拉面的 per-卡分配不同：地区分身由 `at_trains` 指定位置、每位抽一张不在该位
    /// 的支援卡，放不下就是该位不出分身而非改去别处，且来源不含友人卡。
    /// 原实现候选卡列表写死 `(0..6)`（人头下标当卡组下标），且先抽卡再查满员。
    #[test]
    fn test_region_clones_per_train_semantics() -> anyhow::Result<()> {
        use crate::rng::StrategyRng;

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        const SEEDS: u64 = 256;
        // 地区 6「中京-力根」：at_trains = [2, 3]，两个训练位各出一个分身
        const REGION_ID: usize = 6;

        let mut game = RamenGame::newgame(102601, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.add_reporter();
        game.deck_can_split = true;

        let cards: Vec<i32> = (0..game.persons.len() as i32)
            .filter(|&i| game.persons[i as usize].person_type == PersonType::Card)
            .collect();
        let non_card: Vec<i32> = (0..game.persons.len() as i32)
            .filter(|&i| game.persons[i as usize].person_type != PersonType::Card)
            .collect();
        let reporter = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Reporter)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("找不到记者"))?;
        println!("训练卡人头={cards:?}，非训练卡人头={non_card:?}，记者={reporter}");

        let mut c = Checks::new();
        c.check(cards.len() == 5, "测试卡组应有 5 张训练卡");

        // 场景 1：两个 at_trains 位各恰好一个分身，来源必须是训练卡
        let mut s1 = (true, true, true);
        let mut same_card_both = 0usize;
        for seed in 0..SEEDS {
            game.base.distribution = vec![vec![]; 5];
            game.distribute_region_clones(REGION_ID, &mut StdRng::seed_from_u64(seed))?;
            if seed == 0 {
                println!("地区分身（空分布，seed 0）: {:?}", game.base.distribution);
            }
            s1.0 &= game.base.distribution[2].len() == 1 && game.base.distribution[3].len() == 1;
            s1.1 &= [0usize, 1, 4].iter().all(|&t| game.base.distribution[t].is_empty());
            s1.2 &= game
                .base
                .distribution
                .iter()
                .flatten()
                .all(|&p| game.persons[p as usize].person_type == PersonType::Card);
            if game.base.distribution[2] == game.base.distribution[3] {
                same_card_both += 1;
            }
        }
        c.check(s1.0, "at_trains 的每个训练位各生成 1 个分身");
        c.check(s1.1, "非 at_trains 的训练位不得出现分身");
        c.check(s1.2, "地区分身来源只能是支援卡，绝不含友人卡 / 理事长 / 记者");
        println!("两个位置抽到同一张卡: {same_card_both}/{SEEDS} 次（规格允许，非缺陷）");

        // 场景 2：力位被 5 个非 NPC 占满 -> 该位跳过、不报错；根位照常出分身
        //
        // 占位的 5 个非 NPC 必须**至少留一张候选卡在外面**：若直接把 5 张候选卡全塞进去，
        // 每张都会先被 `can_place_clone` 的「该位已有本体」挡掉，容量判定一次都跑不到，
        // 把 `non_npc_count >= 5` 整段删掉这个测试照样绿（假绿）。
        // 这里放 4 张卡 + 记者，第 5 张卡不在该位、能走到容量判定上被拒。
        let mut s2 = (true, true, true);
        let full_board: Vec<i32> = cards[..4].iter().copied().chain(std::iter::once(reporter)).collect();
        for seed in 0..SEEDS {
            game.base.distribution = vec![vec![]; 5];
            game.base.distribution[2] = full_board.clone();
            let r = game.distribute_region_clones(REGION_ID, &mut StdRng::seed_from_u64(seed));
            s2.0 &= r.is_ok();
            s2.1 &= game.base.distribution[2].len() == 5; // 满员位没有被塞进第 6 个
            s2.2 &= game.base.distribution[3].len() == 1; // 另一位不受影响
        }
        c.check(s2.0, "满员位跳过不得返回 Err（规格内跳过，不是失败）");
        c.check(s2.1, "满 5 个非 NPC 的训练位不得再加分身");
        c.check(s2.2, "某位跳过不影响 at_trains 的其他训练位");

        // 场景 3：同一张卡的本体已在该位时不得重复，**且必须抽到别的卡**
        //
        // 只断言「cards[0] 没变成两个」是假绿：退回「先从 5 张卡抽一次、can_place 失败就
        // 跳过该位」时，有 1/5 的种子抽中 cards[0] 自己 → 该位干脆不出分身，而
        // cards[0] 的计数仍是 1，断言照过。地区路径「先过滤再抽」的实际修复正是
        // 「该位还有合法卡时必须放下」，必须由 s3.1 / s3.2 锁住。
        let mut s3 = (true, true, true);
        for seed in 0..SEEDS {
            game.base.distribution = vec![vec![]; 5];
            game.base.distribution[2] = vec![cards[0]];
            game.distribute_region_clones(REGION_ID, &mut StdRng::seed_from_u64(seed))?;
            let d = &game.base.distribution[2];
            s3.0 &= d.iter().filter(|&&p| p == cards[0]).count() == 1;
            s3.1 &= d.len() == 2;
            s3.2 &= d
                .iter()
                .filter(|&&p| p != cards[0])
                .all(|&p| game.persons[p as usize].person_type == PersonType::Card);
            if seed == 0 {
                println!("场景 3（力位已有 cards[0]，seed 0）: {d:?}");
            }
        }
        c.check(s3.0, "同一训练位不得同时存在本体与分身");
        c.check(s3.1, "该位仍有合法卡时必须放下一个分身（回退「先抽再跳过」会 1/5 空放）");
        c.check(s3.2, "补上的那个人头必须是支援卡");

        // 场景 4a：注入 rule_master 后完全不消耗父流，且与父流此前消耗次数无关
        //
        // 地区分身在吃面落地时执行，父流 counter 取决于本回合此前的动作与事件。
        // 按 (rule_master, turn, TAG) 派生后这条耦合被切断——这是 MCTS 配对对齐的前提。
        {
            use crate::rng::StrategyRng;
            use rand::RngCore;

            let mut g = RamenGame::newgame(102601, &TEST_DECK, TEST_INHERIT)?;
            g.add_friend_and_npcs()?;
            g.deck_can_split = true;
            g.set_rule_master(0x5EED_9999);

            let mut zero_draw = true;
            let mut outs = Vec::new();
            for pre in [0usize, 1, 9] {
                g.base.distribution = vec![vec![]; 5];
                let mut parent = StrategyRng::new(0xFEED_0001);
                for _ in 0..pre {
                    let _ = parent.next_u64();
                }
                let before = parent.counter();
                g.distribute_region_clones(REGION_ID, &mut parent)?;
                let used = parent.counter() - before;
                println!("地区分身（注入 rule_master，父流预消耗 {pre}）: {:?}，本次消耗 {used} 次",
                    g.base.distribution);
                zero_draw &= used == 0;
                outs.push(g.base.distribution.clone());
            }
            c.check(zero_draw, "注入 rule_master 后地区分身完全不消耗父策略流");
            c.check(
                outs.windows(2).all(|w| w[0] == w[1]),
                "父流此前消耗多少次都不影响地区分身结果（CRN 对齐的前提）"
            );
        }

        // 场景 4b：未注入 rule_master 的旧路径回退从父流 fork，消耗恰好 1 次
        let mut all_one = true;
        for (name, board) in [
            ("空分布", vec![vec![]; 5]),
            ("力位满员（4 卡 + 记者，第 5 张卡走到容量判定）", {
                let mut d = vec![vec![]; 5];
                d[2] = full_board.clone();
                d
            })
        ] {
            game.base.distribution = board;
            let mut parent = StrategyRng::new(0xC0FF_EE00);
            game.distribute_region_clones(REGION_ID, &mut parent)?;
            println!("地区分身父流消耗: {name} -> {} 次", parent.counter());
            all_one &= parent.counter() == 1;
        }
        c.check(all_one, "未注入 rule_master 时回退从父流 fork，消耗恰好 1 次（含满员跳过的局面）");

        // 场景 5：id < 5 的地区不触发分身
        game.base.distribution = vec![vec![]; 5];
        game.distribute_region_clones(4, &mut StdRng::seed_from_u64(7))?;
        c.check(
            game.base.distribution.iter().flatten().count() == 0,
            "地区 id < 5 不应生成任何分身"
        );

        c.finish()
    }

    /// 探测 trainer：数 `select_action` 里 RegionSelect 动作出现的次数，以及当时的 `game.stage`
    struct RegionProbe {
        /// 任意阶段收到地区动作的次数
        region_calls: std::cell::Cell<usize>,
        /// `stage == Begin` 时收到地区动作的次数
        begin_region_calls: std::cell::Cell<usize>
    }

    impl Trainer<RamenGame> for RegionProbe {
        fn select_action(
            &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], _rng: &mut StdRng
        ) -> Result<usize> {
            if actions.iter().any(|a| matches!(a.operation, Operation::RegionSelect(_))) {
                self.region_calls.set(self.region_calls.get() + 1);
                if game.stage == RamenStage::Begin {
                    self.begin_region_calls.set(self.begin_region_calls.get() + 1);
                }
            }
            Ok(0)
        }

        fn select_choice(
            &self, _game: &RamenGame, _choices: &[Vec<EventChoice>], _rng: &mut StdRng
        ) -> Result<usize> {
            Ok(0)
        }

        fn select_event_choice(
            &self,
            _game: &RamenGame,
            _event: &crate::gamedata::EventData,
            _choices: &[Vec<EventChoice>],
            _rng: &mut StdRng
        ) -> Result<usize> {
            Ok(0)
        }
    }

    /// 推进到 turn 2 的 `Begin`
    fn advance_to_turn2_begin(game: &mut RamenGame, trainer: &impl Trainer<RamenGame>, rng: &mut StdRng) -> Result<()> {
        game.run_stage(trainer, rng)?;
        while game.next() {
            if game.turn() == 2 && game.stage == RamenStage::Begin {
                return Ok(());
            }
            game.run_stage(trainer, rng)?;
        }
        anyhow::bail!("未能推进到 turn 2 Begin")
    }

    /// turn 2 阶段序列严格为 Begin → RegionSelect → BeginAfterRegionSelect → Distribute
    #[test]
    fn test_turn2_stage_sequence() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(1);
        let probe = RegionProbe {
            region_calls: std::cell::Cell::new(0),
            begin_region_calls: std::cell::Cell::new(0)
        };
        let mut rng = StdRng::seed_from_u64(1);
        advance_to_turn2_begin(&mut game, &probe, &mut rng)?;

        let mut c = Checks::new();
        let mut seq = vec![format!("{:?}", game.stage)];
        game.run_stage(&probe, &mut rng)?;
        c.check(game.turn() == 2, "Begin 前半后回合仍为 2");
        c.check(probe.begin_region_calls.get() == 0, "run_begin 不再内联选地区");
        Game::next(&mut game);
        seq.push(format!("{:?}", game.stage));
        c.check(game.stage == RamenStage::RegionSelect, "Begin 后是 RegionSelect");
        c.check(game.turn() == 2, "进入 RegionSelect 时回合仍为 2");

        let actions = game.list_actions()?;
        println!("turn2 RegionSelect 候选={}", actions.len());
        c.check(
            actions
                .iter()
                .all(|a| matches!(a.operation, Operation::RegionSelect(_))),
            "list_actions 在 RegionSelect 只给地区动作"
        );
        c.check(actions.len() == 10, "第 1 年 C(5,3)=10");

        game.run_stage(&probe, &mut rng)?;
        c.check(probe.region_calls.get() == 1, "地区选择只走 trainer 一次");
        c.check(probe.begin_region_calls.get() == 0, "Begin 阶段从未收到地区动作");
        c.check(
            game.ramen.yearly_selected_regions[0] != [0, 0, 0],
            "第 1 年归档已写入"
        );
        Game::next(&mut game);
        seq.push(format!("{:?}", game.stage));
        c.check(
            game.stage == RamenStage::BeginAfterRegionSelect,
            "RegionSelect 后是 BeginAfterRegionSelect"
        );
        c.check(game.turn() == 2, "地区选择后回合仍为 2");

        let persons_before_suffix = game.persons.len();
        let special_before_suffix = game.ramen.special_feeling;
        let y1_before_suffix = game.ramen.yearly_selected_regions[0];
        let event_before = game.event.map(|e| e.counter()).unwrap_or(0);
        game.run_stage(&probe, &mut rng)?;
        Game::next(&mut game);
        seq.push(format!("{:?}", game.stage));
        println!("turn2 序列: {seq:?}");
        println!(
            "人头 {}→{} 隐藏风味 {}→{} 事件流 {}→{} 归档 {:?}",
            persons_before_suffix,
            game.persons.len(),
            special_before_suffix,
            game.ramen.special_feeling,
            event_before,
            game.event.map(|e| e.counter()).unwrap_or(0),
            game.ramen.yearly_selected_regions[0]
        );
        let seq_text = seq.join(" → ");
        c.check(
            seq_text == "Begin → RegionSelect → BeginAfterRegionSelect → Distribute",
            "turn 2 阶段序列严格为 Begin → RegionSelect → BeginAfterRegionSelect → Distribute"
        );
        c.check(game.turn() == 2, "后半段结束后回合仍为 2");
        c.check(game.persons.len() == persons_before_suffix, "人头初始化只发生一次");
        c.check(
            game.ramen.special_feeling == special_before_suffix,
            "诀窍/隐藏风味初始化只发生一次（后半段不再 init）"
        );
        c.check(
            game.ramen.yearly_selected_regions[0] == y1_before_suffix,
            "第 1 年归档只写一次"
        );
        c.check(
            game.event.map(|e| e.counter()).unwrap_or(0) > event_before,
            "Begin 后半段事件链确实执行（事件流被消耗）"
        );
        c.finish()
    }

    /// 非 turn 2 的回合阶段序列不含 `BeginAfterRegionSelect`
    #[test]
    fn test_non_turn2_has_no_begin_after_region_select() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut c = Checks::new();
        for turn in [0, 1, 3, 24] {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.base.turn = turn;
            game.stage = RamenStage::Begin;
            Game::next(&mut game);
            println!("turn={turn} Begin.next → {:?}", game.stage);
            c.check(
                game.stage == RamenStage::Distribute,
                &format!("turn {turn} Begin 下一阶段是 Distribute")
            );
            c.check(
                game.stage != RamenStage::BeginAfterRegionSelect,
                &format!("turn {turn} 不含 BeginAfterRegionSelect")
            );
        }
        for turn in [23, 47] {
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.base.turn = turn;
            game.stage = RamenStage::RegionSelect;
            Game::next(&mut game);
            println!("turn={turn} RegionSelect.next → turn={} {:?}", game.turn(), game.stage);
            c.check(game.turn() == turn + 1, &format!("turn {turn} RegionSelect 推进回合"));
            c.check(
                game.stage == RamenStage::Begin,
                &format!("turn {turn} RegionSelect 下一阶段是下一回合 Begin")
            );
        }
        c.finish()
    }

    /// 选面预演借用训练候选时保留普通训练和必赛规则，且不改动阶段或 pending。
    #[test]
    fn test_train_actions_independent_of_selection_stage() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.stage = RamenStage::RamenSelect;
        game.ramen.pending_ramen = Some(0);
        game.ramen.pending_special_targets = [1, 0, 0];
        let mut checks = Checks::new();
        let actions = game.list_train_actions();
        checks.check(
            actions.iter().filter(|a| matches!(a.operation, Operation::Train(_))).count() == 5
                && actions.iter().any(|a| a.operation == Operation::Rest),
            "普通回合的训练候选包含五训练位和休息"
        );
        game.base.turn = 11;
        let race = game.list_train_actions();
        println!("训练候选普通={actions:?}，必赛={race:?}");
        checks.check(race.len() == 1 && race[0].operation == Operation::Race, "必赛回合仅允许比赛");
        checks.check(
            game.stage == RamenStage::RamenSelect && game.ramen.pending_ramen == Some(0)
                && game.ramen.pending_special_targets == [1, 0, 0],
            "候选生成保留原阶段与 pending"
        );
        checks.finish()
    }

    /// 第 3 年 `fixed` 策略 `list_actions` 必须是单候选，第 1 年不受影响
    #[test]
    fn test_year3_fixed_list_actions_single_candidate() -> Result<()> {
        use crate::{game::ramen::action::region_select_combos, gamedata::RamenRegionStrategy};
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        // 策略与 fixed 表**显式传参**，不经全局配置。
        // `init_global_with_config` 幂等：globals 已初始化时直接返回 Ok(()) 并丢弃
        // 传入 config，测试并行跑在同一进程里，谁先 init_global 谁说了算。
        // 早先靠它设置 Fixed 的测试其实一直在对着默认配置空转。
        let fixed = [[10usize, 15, 19]];
        let mut c = Checks::new();

        let y1 = region_select_combos(0, RamenRegionStrategy::Fixed, Some(&fixed))?;
        println!("Fixed 策略下第 1 年组合数={}", y1.len());
        c.check(y1.len() == 10, "第 1 年 Fixed 不生效，仍枚举 C(5,3)=10");

        let y2 = region_select_combos(1, RamenRegionStrategy::Fixed, Some(&fixed))?;
        println!("Fixed 策略下第 2 年组合数={}", y2.len());
        c.check(y2.len() == 10, "第 2 年 Fixed 不生效，仍枚举 C(5,3)=10");

        let y3_all = region_select_combos(2, RamenRegionStrategy::All, None)?;
        println!("All 策略下第 3 年组合数={}", y3_all.len());
        c.check(y3_all.len() == 120, "第 3 年 All 枚举 C(10,3)=120");

        let y3_fixed = region_select_combos(2, RamenRegionStrategy::Fixed, Some(&fixed))?;
        println!("Fixed 策略下第 3 年组合={y3_fixed:?}");
        c.check(y3_fixed.len() == 1, "第 3 年 Fixed 必须单候选直达，不能恢复 120 组合");
        c.check(y3_fixed.first() == Some(&[10, 15, 19]), "单候选就是 ramen_region_fixed[0]");

        // fixed 表缺失 / 为空都必须报错，不能静默回退成 120 枚举
        let missing = region_select_combos(2, RamenRegionStrategy::Fixed, None);
        println!("第 3 年 Fixed 但未设置 fixed 表: {missing:?}");
        c.check(missing.is_err(), "fixed 表缺失时报错");
        let empty: [[usize; 3]; 0] = [];
        let empty_res = region_select_combos(2, RamenRegionStrategy::Fixed, Some(&empty));
        println!("第 3 年 Fixed 但 fixed 表为空: {empty_res:?}");
        c.check(empty_res.is_err(), "fixed 表为空时报错");

        c.finish()
    }

    #[test]
    fn test_super_ramen_turn_no_angle_tag() -> Result<()> {
        // 超级拉面回合（72-77）：训练诀窍角标照常抽签但不落库（None ⇒ 训练
        // 不产生诀窍槽）——与在线协议角标全 0 一致（2026-09，见 issues.md）。
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(3);

        // 超级拉面、非比赛回合：跑 distribute 分支（而非 reset_distribution）
        game.base.turn = 74;
        println!("turn74 是否比赛回合: {}", game.is_race_turn());
        assert!(!game.is_race_turn(), "turn74 不应是比赛回合");
        game.run_distribute(&mut rng)?;
        println!(
            "t74 角标={:?}（期望 None）分布非空={}",
            game.ramen.train_feeling_type,
            game.base.distribution.iter().any(|d| !d.is_empty())
        );
        assert!(game.ramen.train_feeling_type.is_none(), "72-77 训练角标不落库");
        assert!(
            game.base.distribution.iter().any(|d| !d.is_empty()),
            "分布分配未被跳过"
        );

        // 对照：普通回合仍分配角标
        game.base.turn = 60;
        game.run_distribute(&mut rng)?;
        println!("t60 角标={:?}（期望 Some）", game.ramen.train_feeling_type);
        assert!(game.ramen.train_feeling_type.is_some(), "普通回合仍分配角标");
        Ok(())
    }

}
