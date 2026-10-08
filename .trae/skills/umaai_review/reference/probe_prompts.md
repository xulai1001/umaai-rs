# 探查子代理提示词模板（固化 · 降级路径）

> **常规路径不用本文件**——bin 已产出 `brief.md`（六问事实预答），SKILL.md 第 3 步
> 一次 Read 即可。本文件只在 **brief.md 缺失**时启用：旧版 exe、或 exe 不可用、
> 或需要 digest.json 里 brief 未覆盖的细节块。
>
> 启用时：主代理**不要临场手写**任务描述——取「公共段」+ 对应「组块」，
> 把 `<DIGEST>` 替换成 digest.json 的**绝对路径**即可派发（一组一代理，Task 工具并行，
> 全部组放**同一条消息一次发出**）。
> 子代理不可用时，主代理按同一组块**串行执行**，替换规则相同，可把多组合并到少数几次命令里。
> 每组取数**同时给出 Python（首选）与 PowerShell（备选）两版**，字段名与方法勿改动。

## 公共段（每条子代理任务描述前必附）

```
你是 umaai_review 单局复盘的探查子代理。任务：从 digest.json 提取指定数据块，
返回**紧凑结论**，供主代理做归因。你不需要写业务代码，只做取数与判读。

【硬性约束】
1. digest.json 是单行紧凑 JSON（200KB 级）。**禁止整读整贴**。取数**优先 Python**：
   把取数写成一个临时 .py，**一次加载后统一打印**（`python -c "..."` 里带嵌套引号会被
   PowerShell 吞掉，复杂取数务必走临时文件）：
       python probe.py
   本机没有 Python 时，用组块里给的 **PowerShell 备选片段**（读文件必须 `-Encoding UTF8`）。
   已知部分会话 PowerShell 执行成功但 stdout 为空，遇到直接换 Python。
2. 结构坑位（务必按此取数，踩了就取到空值或漏数）：
   - `timeline` / `execution` 块**直接是行数组**，不是带 rows 字段的对象
     （Python `d['execution']['rows']` / PS `$d.execution.rows` 都取到空）
   - `luck.series` 每项字段是 `{turn, seq, total_luck}`，**没有 value 字段**
   - `luck.flagged_turns` 是对象数组 `{turn, reason}`，不是数字数组
   - `execution` 行的一致性字段名是 `matches`（布尔或 null），**不是 `match`**；
     PS 里判 null 要用 `$null -ne $_.matches`，`-eq $true` 判一致
   - **`rainbow_positions` 含 0（速位）时，PowerShell 会把单元素数组解包成标量 0，
     `[bool]0 = False`**——过滤彩圈**必须**写 `@($_.rainbow_positions).Count -gt 0`，
     写成 `$_.rainbow_positions -and ...` 会**静默漏掉速位彩圈**（game3101 t63 实测：
     汇总 1 个、明细 0 条，正是此坑）。Python 无此问题，直接判 list 非空。
   - `clones.region_per_turn` 先按彩圈位非空过滤再取字段，勿全量 dump
   - `race_count` / `raceHistory` 在开局快照（t0-t2）可能残留上一局数据，t3 起清零，
     比赛计数从年界稳定后读
3. 机制口径（叙事用，勿当成 bug）：
   - `turn_delta` 是「局面期望终局分」变化，**不等于**本回合属性增量；同回合多段 Δ 以
     **回合合计**为基本观察单位
   - 年界 [24,48,72] / 继承 [30,54] / RMJ 结算 / 开局第 1 年地区选择属**程序性波动**，归因降级或跳过；
     turn 72 双属性（既标记为年界波动，也算进超级拉面期）
   - 超级拉面期 = `turn >= 72`；输赛**不会**掉干劲；turn 73/75/77 固定有 URA 目标赛，按胜利处理
   - 五维叙事报**显示值**（`timeline.five_status_display`），评分与运气分不受显示值换算影响
   - `total_luck_end` `< -2000` 才判「这局运气差」（保守线）；旧局包（brief §0 标了无 keyEvents）曲线系统性略偏低
4. 返回格式：只回**紧凑结论**——用「回合号 + 数值 + 一句判断」的短句或小表格，
   **禁止贴原始 JSON、禁止整段 dump**，控制在 30 行以内。数据缺失就直说缺哪块。

【你负责的数据块与要回答的问题】
<组块内容>
```

## 组 1 · `meta` + `context`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
m = d['meta']
for k in ('game', 'uma_id', 'uma_name', 'start_turn', 'mid_entry', 'end_reason',
          'snapshots', 'decision_rows', 'total_luck_end', 'final_score', 'rank'):
    print(k, '=', m.get(k))
for c in m.get('deck', []):
    print(f"{c['card_id']}|{c['name']}|type={c['card_type']}|lb={c['limit_break']}")
for c in d['context']['criteria']:
    print('-', c)
print(d['context']['luck_formula'])
print(d['context']['region_names'])

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
$d.meta | Select-Object game,uma_id,uma_name,start_turn,mid_entry,end_reason,snapshots,decision_rows,total_luck_end,final_score,rank | Format-List
$d.meta.deck | ForEach-Object { "$($_.card_id)|$($_.name)|type=$($_.card_type)|lb=$($_.limit_break)" }
$d.context.criteria | ForEach-Object { $_ }
$d.context.luck_formula
$d.context.region_names | ConvertTo-Json -Compress
（card_type：0=速 1=耐 2=力 3=根 4=智 5=友人）

回答：马娘与卡组构成 / 终局评分与等级 / 结束方式与是否中途接续 /
终局运气分及其定性（< -2000 才算运气差）/ 归因口径前提有哪几条。

返回：卡组一行、评分与运气分一行、结束方式一行、口径前提 3-4 条短句。
```

## 组 2 · `timeline`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
tl = d['timeline']
print('rows=', len(tl))
if tl:
    print('首行:', json.dumps(tl[0], ensure_ascii=False)[:400])   # 先看清字段名，再按需取
    print('末行:', json.dumps(tl[-1], ensure_ascii=False)[:400])
print('vital 最低:', min((r['vital'] for r in tl), default=None))
print('vital <= 35:', [f"t{r['turn']}/s{r['seq']}={r['vital']}" for r in tl if r['vital'] <= 35])
prev = None                       # 干劲变化点（只在值变化时打印）
for r in tl:
    if prev is None or prev != r['motivation']:
        print(f"t{r['turn']}/s{r['seq']} mot={r['motivation']}")
    prev = r['motivation']
TURNS = [0, 24, 48, 72, 77]       # 回合末快照；往这里补需要细看的回合号
last = {}
for r in tl:
    last[r['turn']] = r           # 后写覆盖 = 回合末
for t in TURNS:
    r = last.get(t)
    if r:
        print(f"t{t} real={r['five_status']} disp={r['five_status_display']} "
              f"lim={r['five_status_limit']} vital={r['vital']}/{r['max_vital']} "
              f"mot={r['motivation']} pt={r['scenario_pt']} fr={r['friend_outgoing_used']}")

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
'rows=' + $d.timeline.Count
'vital最低=' + ($d.timeline | Measure-Object vital -Minimum).Minimum
$d.timeline | Where-Object { $_.vital -le 35 } | ForEach-Object { "t$($_.turn)/s$($_.seq)=$($_.vital)" }
$prev=$null
$d.timeline | ForEach-Object { if ($null -eq $prev -or $prev -ne $_.motivation) { "t$($_.turn)/s$($_.seq) mot=$($_.motivation)" }; $prev=$_.motivation }
$last=@{}; $d.timeline | ForEach-Object { $last[[int]$_.turn] = $_ }   # 后写覆盖 = 回合末
foreach ($t in @(0,24,48,72,77)) { $r=$last[$t]; if ($r) {
  "t$t real=[$($r.five_status -join ',')] disp=[$($r.five_status_display -join ',')] lim=[$($r.five_status_limit -join ',')] vital=$($r.vital)/$($r.max_vital) mot=$($r.motivation) pt=$($r.scenario_pt) fr=$($r.friend_outgoing_used)" } }

回答：五维显示值逐年走势（哪个维何时触顶）/ 体力低点落在哪些回合、最低多少 /
干劲掉落在哪些回合、由什么动作或事件回升 / 剧本 PT 与友人出行次数的进度 /
是否有属性触顶。

返回：五维逐年一行一条、体力低点分条、干劲变化点分条（写清恢复动作）、触顶回合各一行。
```

## 组 3 · `luck`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
lk = d['luck']
print('count=', len(lk['series']))
print('序列:', ' '.join(f"{p['turn']}/{p['seq']}:{p['total_luck']:.0f}" for p in lk['series']))
print('top_gain:', json.dumps(lk['top_gain'], ensure_ascii=False))   # [{turn, delta, segments}]
print('top_loss:', json.dumps(lk['top_loss'], ensure_ascii=False))
print('flagged:', ' '.join(f"{f['turn']}:{f['reason']}" for f in lk['flagged_turns']))
print('raw_delta_stats:', json.dumps(lk['raw_delta_stats'], ensure_ascii=False))

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
'count=' + $d.luck.series.Count
$d.luck.series | ForEach-Object { "$($_.turn)/$($_.seq):$([math]::Round($_.total_luck))" }
$d.luck.top_gain | ForEach-Object { $_ | ConvertTo-Json -Compress }
$d.luck.top_loss | ForEach-Object { $_ | ConvertTo-Json -Compress }
$d.luck.flagged_turns | ForEach-Object { "$($_.turn):$($_.reason)" }
$d.luck.raw_delta_stats | ConvertTo-Json -Compress

回答：series 按年分段（第 1 年 / 第 2 年 / 第 3 年 / 超拉期）的起止值与走势定性 /
top_gain 与 top_loss 各自是几个回合、每回合合计 Δ 多少、segments 怎么分段 /
其中哪些回合被 flagged 标记、reason 是什么 / 终局累计值。

返回：分段走势 4-5 条（写清起止回合与数值）、正负极值各一行（回合 + Δ + 是否 flagged）、
flagged 回合一行汇总。
```

## 组 4 · `decisions`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
# 自行从 luck 取极值回合号（不依赖别的子代理）
TURNS = [t['turn'] for t in d['luck']['top_gain']] + [t['turn'] for t in d['luck']['top_loss']]
# 续：把关键回合（年界 / 继承 / 超拉期）补进 TURNS
for t in dict.fromkeys(TURNS):                     # 去重保序
    print(f'=== turn {t} ===')
    for r in (x for x in d['decisions'] if x['turn'] == t):
        ch = r.get('chosen') or {}
        print(f" s{r['seq']} {r['stage']}/{r['kind']} chosen=[{ch.get('desc')}] "
              f"action_luck={ch.get('action_luck')} tdisp={r.get('t_n_display')} "
              f"delta={r.get('turn_delta')} chain={r.get('chain_len')}")
        for c in r.get('candidates', []):
            print(f"    #{c['rank']} [{c['desc']}] score={c.get('score')} gap={c.get('gap_to_best')}")

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
$g=@($d.luck.top_gain | ForEach-Object { $_.turn }); $l=@($d.luck.top_loss | ForEach-Object { $_.turn })
foreach ($t in @($g + $l)) { '=== turn ' + $t + ' ==='
  $d.decisions | Where-Object { $_.turn -eq $t } | ForEach-Object {
    " s$($_.seq) $($_.stage)/$($_.kind) chosen=[$($_.chosen.desc)] action_luck=$($_.chosen.action_luck) tdisp=$($_.t_n_display) delta=$($_.turn_delta) chain=$($_.chain_len)"
    $_.candidates | ForEach-Object { "    #$($_.rank) [$($_.desc)] score=$($_.score) gap=$($_.gap_to_best)" } } }

坑位：`candidates` 每项是 `{rank, desc, score, n, gap_to_best}`；`chosen` 是 `{idx, desc, action_luck}`；
`turn_delta` 在本回合没有独立决策行时会是 **null**，此时该回合的 Δ 只存在于 `luck.series` 或
`top_gain/top_loss` 的 `segments` 里，要去那边取。

回答：每个极值回合 AI 选中了哪一手、第二名差多少分、该回合有几段 Δ、各段正负如何 /
同回合多段是否符号翻转 / 哪些回合的选项与属性上限有关。

返回：每个极值回合一段（回合号 + 选中项 + 主 Δ（回合合计）+ 一句判断），
不用列全部候选，只列与选中项差距近的 2-3 个。
```

## 组 5 · `execution`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
ex = d['execution']
comparable = sum(1 for r in ex if r.get('matches') is not None)
matched = sum(1 for r in ex if r.get('matches') is True)
print(f'comparable={comparable} matched={matched} mismatch={comparable - matched}')
for r in ex:
    if r.get('matches') is False:
        print(f"t{r['turn']} {r['stage']} ai=[{r['ai_choice']}] actual=[{r['actual_action']}] "
              f"{json.dumps(r['evidence'], ensure_ascii=False)}")

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
'comparable=' + @($d.execution | Where-Object { $null -ne $_.matches }).Count + ' matched=' + @($d.execution | Where-Object { $_.matches -eq $true }).Count
$d.execution | Where-Object { $_.matches -eq $false } | ForEach-Object {
  "t$($_.turn) $($_.stage) ai=[$($_.ai_choice)] actual=[$($_.actual_action)] $($_.evidence | ConvertTo-Json -Compress)" }

回答：一致率是多少（可比 / 一致 / 偏离三个数）/ 每条偏离的回合、AI 建议、推断实际动作、
落地证据（五维增量与体力增量）/ 哪些偏离涉及属性上限或体力阈值边界（这类判读会失准，标存疑）/
是否有「继承混合」行，这类不参与一致率。

返回：一致率一行、每条偏离一行（回合 + 建议 + 实际 + 证据摘要 + 是否存疑）。
```

## 组 6 · `schedule` + `coverage`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
s = d['schedule']
print('必赛回合:', ', '.join(map(str, s.get('mandatory_turns', []))))
for n in s.get('notes', []):
    print('注记:', n)
print('free_races:', json.dumps(s.get('free_races'), ensure_ascii=False))
print('coverage:', json.dumps(d['coverage'], ensure_ascii=False))

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
($d.schedule.mandatory_turns) -join ','
$d.schedule.notes | ForEach-Object { $_ }
$d.schedule.free_races | ConvertTo-Json -Compress -Depth 4
$d.coverage | ConvertTo-Json -Compress -Depth 4
（coverage 是单个对象，不是数组：`{skip:{by_reason:{...}}, no_emit, unparsed, parse_error}`）

回答：必赛回合列表是哪些 / 有没有必赛未跑赢 / 自由比赛区间与要求次数（注意 freeRaces 有
null / 对象 / 数组三种形态，为空时如实说「该马娘没有自由比赛区间数据」）/ 区间外自选比赛
落在哪些回合 / skip 行的分原因计数 / 有没有解析失败。

返回：必赛一行、自选赛一行（区分区间内与区间外）、skip 汇总一行、异常项一行。
```

## 组 7 · `inherit`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
print('inherit:', json.dumps(d.get('inherit'), ensure_ascii=False))
# 字段：turns（继承回合号）/ contributions（各次五维贡献）
#       reference_value（参考值）/ dev_note（参考值口径说明）
last = {}                         # 需要核实剥离是否干净时，取继承回合前后末快照的五维差
for r in d['timeline']:
    last[r['turn']] = r
for t in (29, 30, 53, 54):
    r = last.get(t)
    if r:
        print(f"t{t} five={r['five_status']}")

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
$d.inherit | ConvertTo-Json -Compress -Depth 4
$last=@{}; $d.timeline | ForEach-Object { $last[[int]$_.turn] = $_ }
foreach ($t in @(29,30,53,54)) { $r=$last[$t]; if ($r) { "t$t five=[$($r.five_status -join ',')]" } }

回答：两次继承（回合 30 / 54）各自的五维贡献、与参考值的偏差（正为偏优、负为偏弱）/
参考值怎么来的（只比五维不含 PT）/ 两次合起来是什么成色。

返回：每次继承一行（回合 + 贡献 + 偏差）、一句总判断（偏优 / 正常 / 偏弱）。
```

## 组 8 · `clones`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
cl = d.get('clones')
if not cl:
    print('clones 块缺失（gamedata 不可用）')
else:
    # 顶层只有 region / super_ramen_clones / region_per_turn，汇总数字在前两个上
    print('地区分身汇总:', json.dumps(cl['region'], ensure_ascii=False))
    print('超级拉面分身汇总:', json.dumps(cl['super_ramen_clones'], ensure_ascii=False))
    # 地区分身逐回合：只取有彩圈的回合（card 是卡序号，不是 card_id）
    for ct in cl.get('region_per_turn', []):
        hits = [c for c in ct.get('cards', []) if c.get('rainbow_positions')]
        if hits:
            print(f"t{ct['turn']}: " + ' ; '.join(
                f"card{c['card']} rainbow={c['rainbow_positions']} "
                f"used={c.get('used')} origin={c.get('origin')}" for c in hits))

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
$d.clones | Select-Object -Property region,super_ramen_clones | ConvertTo-Json -Compress -Depth 4
# ⚠ 必须用 @(...).Count -gt 0：写成 $_.rainbow_positions -and ... 会漏掉速位（position 0）
$d.clones.region_per_turn | ForEach-Object {
  $rc=@($_.cards | Where-Object { @($_.rainbow_positions).Count -gt 0 })
  if ($rc.Count -gt 0) { "t$($_.turn): " + (($rc | ForEach-Object { "card$($_.card) rainbow=[$($_.rainbow_positions -join ',')] used=$($_.used) origin=$($_.origin)" }) -join ' ; ') } }

坑位：`region_per_turn[].cards[]` 是 `{card, positions, rainbow_positions[, origin, used]}`，
其中 **`origin` 与 `used` 只在彩圈项上出现**，非彩圈卡只有 card / positions；
`origin` 取值 `luck`（随机有效增加）或规则来源，`used` 为是否被当回合训练吃到。
**明细条数必须与汇总的「落得意位」数一致**；对不上就是取数漏了（PS 的 position 0 坑最典型），
不要当成数据缺口。

回答：地区分身（turn<72）新增几个、落得意位（真彩圈）几个、被训练几个、来源
随机 / 规则各几个 / 超级拉面分身（turn>=72，只统计训练卡）同样一组数 / 地区分身彩圈分别落在
哪些回合、有没有吃到（used 字段）/ 超级拉面分身只报汇总（机制保证落位，没吃到不算亏）。

返回：地区分身 / 超级拉面分身各一行汇总、地区分身逐次彩圈分条（回合 + 卡 + 位 + used）。
```

## 组 9 · `findings`

```
取数（Python · 首选）：
import json
d = json.load(open(r'<DIGEST>', encoding='utf-8'))
for f in d.get('findings', []):
    print(f"[{f['severity']}] {f['type']} t{f['turn']}: {f['evidence']}")

取数（PowerShell · 备选）：
$d = Get-Content '<DIGEST>' -Raw -Encoding UTF8 | ConvertFrom-Json
$d.findings | ForEach-Object { "[$($_.severity)] $($_.type) t$($_.turn): $($_.evidence)" }

回答：bin 自动命中了哪些检查项（type / turn / evidence / severity）/ 哪些是执行偏离、
哪些是坏手法或机制类结论时 / 严重度为 warn 的有哪几条。

返回：按 type 分类分条列出（回合 + 证据摘要 + severity），warn 项单独点出。
```
