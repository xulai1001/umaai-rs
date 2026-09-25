//! 客户端拉面决策器的装配：按 `ramen_trainer_policy` 决定动作决策由谁负责
//!
//! | `ramen_trainer_policy` | 执行 | 额外显示 | 速度 |
//! |---|---|---|---|
//! | `mcts`（默认） | 既有搜索逻辑 | 无 | 与上游相同 |
//! | `mcts_nn_hint` | 既有搜索逻辑（与 `mcts` 完全一致） | 每个动作决策下一行网络推荐 | 与 `mcts` 基本相同 |
//! | `nn` | 网络直接决策，不做任何搜索 | 无 | 远快于搜索，分数更低；只适用于训练覆盖的卡组构成 |
//!
//! 事件选项在三种取值下都走手写策略。网络不进搜索：搜索内部模拟仍是手写 rollout。
//!
//! 未开 `onnx` feature 却选了网络、模型或旁车缺失、维度不符时启动即报错，
//! 不会降级成搜索继续跑。

use anyhow::{Result, bail};
use rand::prelude::StdRng;
use umasim::{
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{EventChoice, EventData, GameConfig, RamenTrainerPolicy},
    output::DecisionInfo,
    trainer::RamenMctsTrainer
};

#[cfg(feature = "onnx")]
mod hint;
#[cfg(feature = "onnx")]
mod whole;
#[cfg(feature = "onnx")]
pub use hint::NnHintTrainer;
#[cfg(feature = "onnx")]
pub use whole::WholeNnTrainer;

/// 客户端实际对局使用的拉面决策器
pub enum RamenClientTrainer {
    /// 默认装配：与上游搜索训练员完全相同
    Mcts(RamenMctsTrainer),
    /// 搜索执行，网络给参考
    #[cfg(feature = "onnx")]
    MctsNnHint(Box<NnHintTrainer>),
    /// 网络直接决策
    #[cfg(feature = "onnx")]
    Nn(Box<WholeNnTrainer>)
}

/// 对每个变体调用同一个 [`Trainer`] 方法
macro_rules! dispatch {
    ($self:ident, $t:ident => $call:expr) => {
        match $self {
            Self::Mcts($t) => $call,
            #[cfg(feature = "onnx")]
            Self::MctsNnHint($t) => $call,
            #[cfg(feature = "onnx")]
            Self::Nn($t) => $call
        }
    };
}

impl RamenClientTrainer {
    /// 与配置取值一致的策略标签（启动时打印）
    pub fn label(&self) -> &'static str {
        match self {
            Self::Mcts(_) => "mcts",
            #[cfg(feature = "onnx")]
            Self::MctsNnHint(_) => "mcts_nn_hint",
            #[cfg(feature = "onnx")]
            Self::Nn(_) => "nn"
        }
    }
}

impl Trainer<RamenGame> for RamenClientTrainer {
    /// 按变体分派
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        dispatch!(self, t => t.select_action(game, actions, rng))
    }

    /// 事件选项（旧接口），按变体分派
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        dispatch!(self, t => t.select_choice(game, choices, rng))
    }

    /// 事件选项（新接口），按变体分派
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        dispatch!(self, t => t.select_event_choice(game, event, choices, rng))
    }

    /// 上一次决策的协议摘要
    fn last_decision(&self) -> Option<DecisionInfo> {
        dispatch!(self, t => t.last_decision())
    }

    /// 上一次决策的评分分解
    fn last_breakdown(&self) -> Option<String> {
        dispatch!(self, t => t.last_breakdown())
    }
}

/// 校验决策策略配置：需要网络的取值必须给出非空的模型路径
///
/// # 错误
///
/// `nn` / `mcts_nn_hint` 下缺模型路径（或路径为空白）时报错。
pub fn validate_trainer_policy(cfg: &GameConfig) -> Result<()> {
    if !cfg.ramen_trainer_policy.needs_model() {
        return Ok(());
    }
    match cfg.ramen_nn_model_path.as_deref() {
        Some(p) if !p.trim().is_empty() => Ok(()),
        _ => bail!(
            "配置缺失：ramen_trainer_policy={policy:?} 需要 ramen_nn_model_path（含同名 .json 旁车）。             不要填 neuralnet_model_path 的温泉模型，两者结构不同",
            policy = cfg.ramen_trainer_policy
        )
    }
}

/// 按配置装配客户端拉面决策器
///
/// `mcts` 由调用方按既有路径构造好（阶段门控、reason sink、verbose 都已设好）。
/// `mcts` 取值下原样返回它；其余取值加载一次网络模型。
///
/// # 错误
///
/// 配置校验不过（见 [`validate_trainer_policy`]）、未开 `onnx` feature 却选了网络、
/// 模型或旁车缺失、维度不符时报错。
pub fn build_client_trainer(cfg: &GameConfig, mcts: RamenMctsTrainer) -> Result<RamenClientTrainer> {
    validate_trainer_policy(cfg)?;
    match cfg.ramen_trainer_policy {
        RamenTrainerPolicy::Mcts => Ok(RamenClientTrainer::Mcts(mcts)),
        _ => build_nn(cfg, mcts)
    }
}

/// 网络取值的实际装配（开了 `onnx` feature）
///
/// # 错误
///
/// 模型或旁车缺失、维度与契约不符、ONNX 图无法转为可运行图时报错。
#[cfg(feature = "onnx")]
fn build_nn(cfg: &GameConfig, mcts: RamenMctsTrainer) -> Result<RamenClientTrainer> {
    use std::path::Path;

    use umasim::trainer::{RamenNnTrainer, SpecialSelectMode};

    let path = cfg.ramen_nn_model_path.as_deref().unwrap_or_default();
    // 与训练评测时相同的口径：自选比赛硬守门开、SpecialSelect 还原到联合决策根
    let nn = RamenNnTrainer::load(Path::new(path))?
        .with_race_shield(true)
        .with_special_mode(SpecialSelectMode::Canonical);
    Ok(match cfg.ramen_trainer_policy {
        RamenTrainerPolicy::MctsNnHint => RamenClientTrainer::MctsNnHint(Box::new(NnHintTrainer::new(mcts, nn))),
        _ => RamenClientTrainer::Nn(Box::new(WholeNnTrainer::new(nn)))
    })
}

/// 未开 `onnx` feature 时选了网络：直接报错
///
/// # 错误
///
/// 恒报错：当前构建没有推理后端。
#[cfg(not(feature = "onnx"))]
fn build_nn(cfg: &GameConfig, _mcts: RamenMctsTrainer) -> Result<RamenClientTrainer> {
    bail!(
        "ramen_trainer_policy={policy:?} 需要启用 onnx feature 的构建：         cargo build --release --features onnx -p umaai",
        policy = cfg.ramen_trainer_policy
    )
}

#[cfg(test)]
mod tests {
    use std::env;

    use umasim::{
        gamedata::{GameConfig, OverrideGameConfig},
        search::SearchConfig,
        utils::get_workspace_root
    };

    use super::*;
    use crate::utils::Checks;

    /// 需要网络的两个取值
    const NN_POLICIES: [RamenTrainerPolicy; 2] = [RamenTrainerPolicy::MctsNnHint, RamenTrainerPolicy::Nn];

    /// 取校验错误文本（通过时为空串）
    fn validate_err(cfg: &GameConfig) -> String {
        validate_trainer_policy(cfg)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    /// 把一段 `game_config.toml` 文本合进仓库的 `default_config.toml`
    ///
    /// 不读用户自己的 `game_config.toml`，合并走与 `load_game_config` 相同的
    /// `OverrideGameConfig::merge`。
    ///
    /// # 错误
    ///
    /// 定位工作区、读默认配置、解析任一侧 TOML 失败时报错（非法取值即由此返回）。
    fn merge_fixture(override_toml: &str) -> Result<GameConfig> {
        let def_path = get_workspace_root()?.join("gamedata").join("default_config.toml");
        let default_config: GameConfig = toml::from_str(&fs_err::read_to_string(&def_path)?)?;
        // 顶层字段必须排在任何 `[表]` 之前；`[config_override]` 是必填表，补一个空表
        let text = format!("{override_toml}\n[config_override]\n");
        let ov: OverrideGameConfig = toml::from_str(&text)?;
        Ok(ov.merge(&default_config))
    }

    /// `mcts` 不要求模型；另外两个取值缺模型路径或路径为空白时报错
    #[test]
    fn test_validate_trainer_policy() -> Result<()> {
        let mut c = Checks::new();
        let base = GameConfig::default_for_init();
        c.check(base.ramen_trainer_policy == RamenTrainerPolicy::Mcts, "代码内默认是 mcts");
        c.check(validate_err(&base).is_empty(), "mcts 不要求模型路径");

        for policy in NN_POLICIES {
            let mut cfg = base.clone();
            cfg.ramen_trainer_policy = policy;
            let e = validate_err(&cfg);
            println!("{policy:?} 缺模型 → {e}");
            c.check(e.contains("ramen_nn_model_path"), &format!("{policy:?} 缺模型时错误指名该字段"));

            cfg.ramen_nn_model_path = Some("   ".to_string());
            c.check(!validate_err(&cfg).is_empty(), &format!("{policy:?} 空白路径等同缺失"));

            cfg.ramen_nn_model_path = Some("saved_models/x.onnx".to_string());
            c.check(validate_err(&cfg).is_empty(), &format!("{policy:?} 有模型路径时通过"));
        }
        c.finish()
    }

    /// 仓库默认配置是 `mcts` 且不指定模型
    #[test]
    fn test_default_config_file_is_mcts() -> Result<()> {
        let mut c = Checks::new();
        let cfg = merge_fixture("")?;
        println!(
            "default_config.toml：ramen_trainer_policy={:?} ramen_nn_model_path={:?}",
            cfg.ramen_trainer_policy, cfg.ramen_nn_model_path
        );
        c.check(cfg.ramen_trainer_policy == RamenTrainerPolicy::Mcts, "默认决策策略是 mcts");
        c.check(cfg.ramen_nn_model_path.is_none(), "默认不指定网络模型路径");
        c.check(validate_err(&cfg).is_empty(), "默认配置通过校验");
        c.finish()
    }

    /// 覆盖层能解析三个合法取值，未知取值在解析阶段被拒绝
    #[test]
    fn test_override_parses_trainer_policy() -> Result<()> {
        let mut c = Checks::new();
        for (text, want) in [
            ("mcts", RamenTrainerPolicy::Mcts),
            ("mcts_nn_hint", RamenTrainerPolicy::MctsNnHint),
            ("nn", RamenTrainerPolicy::Nn)
        ] {
            let cfg = merge_fixture(&format!("ramen_trainer_policy = \"{text}\""))?;
            println!("{text} → {:?}", cfg.ramen_trainer_policy);
            c.check(cfg.ramen_trainer_policy == want, &format!("{text} 解析为 {want:?}"));
        }

        let only_path = merge_fixture("ramen_nn_model_path = \"saved_models/x.onnx\"")?;
        c.check(
            only_path.ramen_trainer_policy == RamenTrainerPolicy::Mcts,
            "只写模型路径不会切换决策策略"
        );

        for bad in ["handwritten", "nn_compare", "neural"] {
            let r = merge_fixture(&format!("ramen_trainer_policy = \"{bad}\""));
            println!("{bad} → {:?}", r.as_ref().err().map(ToString::to_string));
            c.check(r.is_err(), &format!("未知取值 {bad} 被拒绝"));
        }
        c.finish()
    }

    /// 模型加载的错误路径：文件缺失 / 旁车缺失 / 旁车维度不符 / 不是合法 ONNX
    ///
    /// 全部必须在装配时报错。前三条在解析 ONNX 之前命中，占位文件即可覆盖。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_model_error_paths() -> Result<()> {
        use crate::utils::{cleanup_test_dir, unique_test_dir};

        env::set_current_dir(get_workspace_root()?)?;
        let mut c = Checks::new();
        let mut cfg = GameConfig::default_for_init();
        cfg.ramen_trainer_policy = RamenTrainerPolicy::Nn;
        let err_of = |cfg: &GameConfig| {
            let mcts = RamenMctsTrainer::new(SearchConfig::new_game_config(&GameConfig::default_for_init()));
            build_client_trainer(cfg, mcts)
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default()
        };

        cfg.ramen_nn_model_path = Some("saved_models/__definitely_missing__.onnx".to_string());
        let e1 = err_of(&cfg);
        println!("模型缺失 → {e1}");
        c.check(e1.contains("不存在"), "模型缺失时报错");

        let dir = unique_test_dir("ramen_nn_model_error_paths")?;
        let placeholder = b"not-a-real-onnx";
        let meta = |input_dim: usize| {
            format!(
                r#"{{"input_dim":{input_dim},"output_dim":245,"value_normalization":{{"center":[0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}}}}"#
            )
        };

        let no_meta = dir.join("no_meta.onnx");
        fs_err::write(&no_meta, placeholder)?;
        cfg.ramen_nn_model_path = Some(no_meta.to_string_lossy().into_owned());
        let e2 = err_of(&cfg);
        println!("旁车缺失 → {e2}");
        c.check(e2.contains("元数据"), "旁车缺失时错误指明模型元数据");

        let bad_dim = dir.join("bad_dim.onnx");
        fs_err::write(&bad_dim, placeholder)?;
        fs_err::write(dir.join("bad_dim.onnx.json"), meta(7))?;
        cfg.ramen_nn_model_path = Some(bad_dim.to_string_lossy().into_owned());
        let e3 = err_of(&cfg);
        println!("维度不符 → {e3}");
        c.check(e3.contains("input_dim"), "维度不符时错误指明 input_dim");

        let not_onnx = dir.join("not_onnx.onnx");
        fs_err::write(&not_onnx, placeholder)?;
        fs_err::write(dir.join("not_onnx.onnx.json"), meta(754))?;
        cfg.ramen_nn_model_path = Some(not_onnx.to_string_lossy().into_owned());
        let e4 = err_of(&cfg);
        println!("非法 ONNX → {e4}");
        c.check(!e4.is_empty(), "文件不是合法 ONNX 时报错");

        cleanup_test_dir(&dir)?;
        c.check(!dir.exists(), "fixture 目录已清理");
        c.finish()
    }

    /// 未开 `onnx` feature 时，需要网络的取值装配即报错
    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_nn_without_onnx_feature_errors() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        let mut c = Checks::new();
        for policy in NN_POLICIES {
            let mut cfg = GameConfig::default_for_init();
            cfg.ramen_trainer_policy = policy;
            cfg.ramen_nn_model_path = Some("saved_models/x.onnx".to_string());
            let mcts = RamenMctsTrainer::new(SearchConfig::new_game_config(&cfg));
            let msg = build_client_trainer(&cfg, mcts)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            println!("{policy:?} 未开 onnx → {msg}");
            c.check(msg.contains("onnx"), &format!("{policy:?}：错误信息指出需要 onnx feature"));
        }
        c.finish()
    }
}
