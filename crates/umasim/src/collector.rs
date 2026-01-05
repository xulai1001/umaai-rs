//! Collector 工具支持（P1）
//!
//! 主要用于 mean-filter 数据生成：
//! - 分片写盘（part_*.bin）
//! - manifest（可恢复、可复现）
//! - scoreMean values 文件（用于精确分位数统计）

use std::{
    ffi::OsStr,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::{
    gamedata::CollectorConfig,
    search::SearchConfig,
    training_sample::{TrainingSample, TrainingSampleBatch},
};

// ============================================================================
// 可复现信息
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSignature {
    pub path: String,
    pub size: u64,
    pub modified_unix: Option<i64>,
    pub hash_fnv1a64: Option<String>,
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hex_u64(v: u64) -> String {
    format!("{:016x}", v)
}

pub fn compute_file_signature(path: &Path, hash: bool) -> Result<FileSignature> {
    let meta = fs_err::metadata(path)
        .with_context(|| format!("读取文件元信息失败: {}", path.display()))?;
    let modified_unix = meta.modified().ok().and_then(|t| {
        t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs() as i64)
    });
    let hash_fnv1a64 = if hash {
        let bytes = fs_err::read(path).with_context(|| format!("读取文件失败: {}", path.display()))?;
        Some(hex_u64(fnv1a64(&bytes)))
    } else {
        None
    };
    Ok(FileSignature {
        path: path.to_string_lossy().to_string(),
        size: meta.len(),
        modified_unix,
        hash_fnv1a64,
    })
}

pub fn compute_text_hash_fnv1a64(text: &str) -> String {
    hex_u64(fnv1a64(text.as_bytes()))
}

pub fn try_get_git_commit(repo_dir: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(repo_dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

// ============================================================================
// Manifest
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestSearchConfig {
    pub search_n: usize,
    pub max_depth: usize,
    pub radical_factor_max: f64,
    pub policy_delta: f64,
    pub use_ucb: bool,
    pub search_group_size: usize,
    pub search_cpuct: f64,
    pub expected_search_stdev: f64,
}

impl ManifestSearchConfig {
    pub fn from_search_config(cfg: &SearchConfig) -> Self {
        Self {
            search_n: cfg.search_n,
            max_depth: cfg.max_depth,
            radical_factor_max: cfg.radical_factor_max,
            policy_delta: cfg.policy_delta,
            use_ucb: cfg.use_ucb,
            search_group_size: cfg.search_group_size,
            search_cpuct: cfg.search_cpuct,
            expected_search_stdev: cfg.expected_search_stdev,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestProgress {
    pub games_run: u64,
    pub games_failed: u64,
    pub candidates: u64,
    pub accepted: u64,
    pub dropped: u64,
    pub dropped_zero_mean: u64,
    pub search_errors: u64,
    pub policy_sum_not_one: u64,

    // ========== Choice（P2）==========

    /// 触发的 decision event 数（候选）
    #[serde(default)]
    pub choice_candidates: u64,
    /// 通过 gate 并落盘的 choice 样本数
    #[serde(default)]
    pub choice_accepted: u64,
    /// 被 gate/target/规则丢弃的 choice 样本数
    #[serde(default)]
    pub choice_dropped: u64,
    /// `sum(choice_target)≈1` 异常计数
    #[serde(default)]
    pub choice_sum_not_one: u64,
    /// choice 样本里 `policy_target` 非 0 的异常计数
    #[serde(default)]
    pub choice_policy_not_zero: u64,
    /// 跳过：choices.len() > CHOICE_DIM
    #[serde(default)]
    pub choice_skipped_too_many_options: u64,
    /// 跳过：chance event（random_choice_prob.is_some）
    #[serde(default)]
    pub choice_skipped_chance_event: u64,
}

fn default_vec_u64_78() -> Vec<u64> {
    vec![0; 78]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestPerTurn {
    pub candidates: Vec<u64>,
    pub accepted: Vec<u64>,
    pub dropped: Vec<u64>,

    // ========== Choice（P2）==========
    #[serde(default = "default_vec_u64_78")]
    pub choice_candidates: Vec<u64>,
    #[serde(default = "default_vec_u64_78")]
    pub choice_accepted: Vec<u64>,
    #[serde(default = "default_vec_u64_78")]
    pub choice_dropped: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestScoreMeanSummary {
    pub min: Option<f64>,
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestPart {
    pub name: String,
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorManifest {
    pub version: u32,
    pub created_at: String,
    pub updated_at: String,

    pub output_dir: String,

    // 复现信息（尽量记录；获取失败则为 None）
    pub git_commit: Option<String>,
    pub config_path: String,
    pub config_hash_fnv1a64: String,
    pub gamedata_sig: Vec<FileSignature>,
    pub model_sig: Option<FileSignature>,

    // 生效参数
    pub collector_config: CollectorConfig,
    pub search_config_effective: ManifestSearchConfig,

    // 进度与统计
    pub progress: ManifestProgress,
    pub per_turn: ManifestPerTurn,
    pub score_mean: ManifestScoreMeanSummary,
    pub parts: Vec<ManifestPart>,
}

impl CollectorManifest {
    pub fn new(
        output_dir: &Path,
        git_commit: Option<String>,
        config_path: &str,
        config_hash_fnv1a64: String,
        gamedata_sig: Vec<FileSignature>,
        model_sig: Option<FileSignature>,
        collector_config: CollectorConfig,
        search_config_effective: &SearchConfig,
    ) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            version: 2,
            created_at: now.clone(),
            updated_at: now,
            output_dir: output_dir.to_string_lossy().to_string(),
            git_commit,
            config_path: config_path.to_string(),
            config_hash_fnv1a64,
            gamedata_sig,
            model_sig,
            collector_config,
            search_config_effective: ManifestSearchConfig::from_search_config(search_config_effective),
            progress: ManifestProgress {
                games_run: 0,
                games_failed: 0,
                candidates: 0,
                accepted: 0,
                dropped: 0,
                dropped_zero_mean: 0,
                search_errors: 0,
                policy_sum_not_one: 0,
                choice_candidates: 0,
                choice_accepted: 0,
                choice_dropped: 0,
                choice_sum_not_one: 0,
                choice_policy_not_zero: 0,
                choice_skipped_too_many_options: 0,
                choice_skipped_chance_event: 0,
            },
            per_turn: ManifestPerTurn {
                candidates: vec![0; 78],
                accepted: vec![0; 78],
                dropped: vec![0; 78],
                choice_candidates: vec![0; 78],
                choice_accepted: vec![0; 78],
                choice_dropped: vec![0; 78],
            },
            score_mean: ManifestScoreMeanSummary {
                min: None,
                mean: None,
                p50: None,
                p90: None,
                p99: None,
            },
            parts: Vec::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs_err::read_to_string(path)
            .with_context(|| format!("读取 manifest 失败: {}", path.display()))?;
        let v: Self = serde_json::from_str(&text)
            .with_context(|| format!("解析 manifest 失败: {}", path.display()))?;
        Ok(v)
    }

    pub fn save_replace(&self, path: &Path) -> Result<()> {
        let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));

        let file = fs_err::File::create(&tmp_path)
            .with_context(|| format!("创建临时 manifest 失败: {}", tmp_path.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)
            .with_context(|| "写入 manifest JSON 失败")?;
        writer.flush().with_context(|| "flush manifest 失败")?;
        writer.get_ref().sync_all().ok(); // 最好努力 sync，但失败不致命

        // Windows 下 rename 通常不覆盖目标文件：采用“替换写”
        if path.exists() {
            fs_err::remove_file(path)
                .with_context(|| format!("删除旧 manifest 失败: {}", path.display()))?;
        }
        fs_err::rename(&tmp_path, path)
            .with_context(|| format!("重命名 manifest 失败: {} -> {}", tmp_path.display(), path.display()))?;
        Ok(())
    }

    pub fn touch_updated_at(&mut self) {
        self.updated_at = Utc::now().to_rfc3339();
    }
}

// ============================================================================
// 分片写盘 + resume
// ============================================================================

#[derive(Debug, Clone)]
pub struct ExistingPartsScan {
    pub parts: Vec<ManifestPart>,
    pub accepted_written: u64,
    pub next_part_index: usize,
}

fn parse_part_index(name: &str) -> Option<usize> {
    // 期望格式：part_000123.bin
    if !name.starts_with("part_") || !name.ends_with(".bin") {
        return None;
    }
    let mid = &name["part_".len()..name.len() - ".bin".len()];
    // 只接受固定 6 位数字，避免把 part_1.bin 之类的文件误识别为合法分片
    // （否则后续会用 format!("part_{:06}.bin") 拼回文件名，导致路径不一致）
    if mid.len() != 6 || !mid.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    mid.parse::<usize>().ok()
}

pub fn scan_part_files(output_dir: &Path) -> Result<Vec<(usize, PathBuf)>> {
    let mut ret = Vec::new();
    if !output_dir.exists() {
        return Ok(ret);
    }
    for entry in fs_err::read_dir(output_dir)
        .with_context(|| format!("读取输出目录失败: {}", output_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(OsStr::to_str) {
            Some(v) => v,
            None => continue,
        };
        if let Some(idx) = parse_part_index(name) {
            ret.push((idx, path));
        }
    }
    ret.sort_by_key(|(i, _)| *i);
    Ok(ret)
}

fn load_part_sample_count(path: &Path) -> Result<usize> {
    let batch = TrainingSampleBatch::load_binary(path.to_string_lossy().as_ref())
        .with_context(|| format!("读取 part 失败: {}", path.display()))?;
    Ok(batch.samples.len())
}

pub fn scan_existing_parts(output_dir: &Path, manifest: Option<&CollectorManifest>) -> Result<ExistingPartsScan> {
    let files = scan_part_files(output_dir)?;
    if files.is_empty() {
        return Ok(ExistingPartsScan {
            parts: Vec::new(),
            accepted_written: 0,
            next_part_index: 0,
        });
    }

    let names_on_disk: Vec<String> = files
        .iter()
        .map(|(idx, _)| format!("part_{:06}.bin", idx))
        .collect();

    if let Some(m) = manifest {
        let names_in_manifest: Vec<String> = m.parts.iter().map(|p| p.name.clone()).collect();
        if names_in_manifest == names_on_disk {
            let accepted_written: u64 = m.parts.iter().map(|p| p.samples as u64).sum();
            let next_part_index = files.last().map(|(i, _)| i + 1).unwrap_or(0);
            return Ok(ExistingPartsScan {
                parts: m.parts.clone(),
                accepted_written,
                next_part_index,
            });
        }
    }

    // 不一致：以文件系统为准，重建 parts（可能较慢，但仅在 resume 不一致时发生）
    let mut parts = Vec::with_capacity(files.len());
    let mut accepted_written: u64 = 0;
    for (idx, path) in files {
        let samples = load_part_sample_count(&path)?;
        parts.push(ManifestPart {
            name: format!("part_{:06}.bin", idx),
            samples,
        });
        accepted_written += samples as u64;
    }

    let next_part_index = parts
        .last()
        .and_then(|p| parse_part_index(&p.name))
        .map(|i| i + 1)
        .unwrap_or(0);

    Ok(ExistingPartsScan {
        parts,
        accepted_written,
        next_part_index,
    })
}

pub struct ShardWriter {
    output_dir: PathBuf,
    manifest_path: PathBuf,
    score_mean_values_path: PathBuf,
    shard_size: usize,
    next_part_index: usize,

    current_shard: Vec<TrainingSample>,
    current_score_means: Vec<f32>,

    pub manifest: CollectorManifest,
}

impl ShardWriter {
    pub fn open_or_create(
        output_dir: &Path,
        manifest_name: &str,
        score_mean_values_name: &str,
        shard_size: usize,
        resume: bool,
        overwrite: bool,
        new_manifest: impl FnOnce() -> Result<CollectorManifest>,
    ) -> Result<(Self, ExistingPartsScan)> {
        if output_dir.exists() {
            if overwrite {
                // 危险操作：清空目录
                for entry in fs_err::read_dir(output_dir)? {
                    let p = entry?.path();
                    if p.is_file() {
                        fs_err::remove_file(&p)?;
                    }
                }
            } else if !resume {
                return Err(anyhow!(
                    "输出目录已存在且未开启 resume/overwrite：{}",
                    output_dir.display()
                ));
            }
        } else {
            fs_err::create_dir_all(output_dir)
                .with_context(|| format!("创建输出目录失败: {}", output_dir.display()))?;
        }

        let manifest_path = output_dir.join(manifest_name);
        let score_mean_values_path = output_dir.join(score_mean_values_name);

        let mut manifest = if manifest_path.exists() && !overwrite {
            CollectorManifest::load(&manifest_path)?
        } else {
            let m = new_manifest()?;
            m
        };

        // 以文件系统为准扫描 part
        let scan = scan_existing_parts(output_dir, Some(&manifest))?;
        manifest.parts = scan.parts.clone();
        manifest.progress.accepted = scan.accepted_written;
        manifest.touch_updated_at();
        manifest.save_replace(&manifest_path)?;

        // 对 score_mean_values.bin 做对齐（允许自动修复）
        align_score_mean_values(&score_mean_values_path, output_dir, &manifest, scan.accepted_written)?;

        let writer = Self {
            output_dir: output_dir.to_path_buf(),
            manifest_path,
            score_mean_values_path,
            shard_size: shard_size.max(1),
            next_part_index: scan.next_part_index,
            current_shard: Vec::with_capacity(shard_size.max(1)),
            current_score_means: Vec::with_capacity(shard_size.max(1)),
            manifest,
        };

        Ok((writer, scan))
    }

    pub fn accepted_written(&self) -> u64 {
        self.manifest.parts.iter().map(|p| p.samples as u64).sum()
    }

    pub fn score_mean_values_path(&self) -> &Path {
        &self.score_mean_values_path
    }

    pub fn push_samples(&mut self, samples: Vec<TrainingSample>) -> Result<()> {
        for sample in samples {
            let score_mean = sample.value_target.first().copied().unwrap_or(0.0);
            self.current_score_means.push(score_mean);
            self.current_shard.push(sample);
            if self.current_shard.len() >= self.shard_size {
                self.flush_shard()?;
            }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        if !self.current_shard.is_empty() {
            self.flush_shard()?;
        }
        Ok(())
    }

    fn flush_shard(&mut self) -> Result<()> {
        let part_name = format!("part_{:06}.bin", self.next_part_index);
        let final_path = self.output_dir.join(&part_name);
        if final_path.exists() {
            return Err(anyhow!("part 文件已存在，疑似 resume/index 计算错误: {}", final_path.display()));
        }

        let tmp_path = PathBuf::from(format!("{}.tmp", final_path.display()));

        // 注意：不要在写盘失败时丢失内存中的 shard（P0/PoC 无所谓，但 P1 长跑必须更稳）
        let shard_samples = std::mem::take(&mut self.current_shard);
        let shard_score_means = std::mem::take(&mut self.current_score_means);
        let batch = TrainingSampleBatch { samples: shard_samples };

        // 先写 part（失败则恢复内存缓冲）
        if let Err(e) = (|| -> Result<()> {
            write_batch_binary(&tmp_path, &batch)?;
            fs_err::rename(&tmp_path, &final_path)
                .with_context(|| format!("重命名 part 失败: {} -> {}", tmp_path.display(), final_path.display()))?;
            Ok(())
        })() {
            // 尽量清理 tmp 文件（忽略错误）
            let _ = fs_err::remove_file(&tmp_path);
            // 恢复内存缓冲，便于上层决定是否重试/退出
            self.current_shard = batch.samples;
            self.current_score_means = shard_score_means;
            return Err(e);
        }

        // part 写入成功后，再追加 score_mean_values（append-only）
        append_f32_values(&self.score_mean_values_path, &shard_score_means)?;

        self.manifest.parts.push(ManifestPart {
            name: part_name,
            samples: batch.samples.len(),
        });
        self.next_part_index += 1;
        Ok(())
    }

    pub fn save_manifest(&mut self) -> Result<()> {
        self.manifest.touch_updated_at();
        self.manifest.save_replace(&self.manifest_path)
    }
}

fn write_batch_binary(path: &Path, batch: &TrainingSampleBatch) -> Result<()> {
    let file = fs_err::File::create(path)
        .with_context(|| format!("创建文件失败: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    bincode::serialize_into(&mut writer, batch).with_context(|| "bincode 写入失败")?;
    writer.flush().with_context(|| "flush part 失败")?;
    writer.get_ref().sync_all().ok();
    Ok(())
}

fn append_f32_values(path: &Path, values: &[f32]) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    let mut file = fs_err::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("打开 score_mean_values 失败: {}", path.display()))?;
    for v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    file.flush().ok();
    file.sync_all().ok();
    Ok(())
}

pub fn load_score_mean_values(path: &Path) -> Result<Vec<f32>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs_err::read(path)
        .with_context(|| format!("读取 score_mean_values 失败: {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "score_mean_values.bin 字节长度不是 4 的倍数: {} (len={})",
            path.display(),
            bytes.len()
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn align_score_mean_values(
    values_path: &Path,
    output_dir: &Path,
    manifest: &CollectorManifest,
    accepted_written: u64,
) -> Result<()> {
    // 若不存在：
    // - accepted_written==0：允许后续写入创建
    // - accepted_written>0 ：必须从 part 重建，否则分位数统计会缺失历史数据
    if !values_path.exists() {
        if accepted_written == 0 {
            return Ok(());
        }

        // values 缺失但已有历史 part：直接重建
        let mut rebuilt: Vec<f32> = Vec::with_capacity(accepted_written as usize);
        for part in &manifest.parts {
            let path = output_dir.join(&part.name);
            let batch = TrainingSampleBatch::load_binary(path.to_string_lossy().as_ref())
                .with_context(|| format!("重建 score_mean_values 读取 part 失败: {}", path.display()))?;
            for s in batch.samples {
                if let Some(v) = s.value_target.first() {
                    rebuilt.push(*v);
                }
            }
        }

        let tmp_path = PathBuf::from(format!("{}.tmp", values_path.display()));
        let mut f = fs_err::File::create(&tmp_path)?;
        for v in &rebuilt {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush().ok();
        f.sync_all().ok();

        // Windows：rename 通常不覆盖；但这里目标本应不存在，仍做一次兜底 remove
        if values_path.exists() {
            fs_err::remove_file(values_path)?;
        }
        fs_err::rename(&tmp_path, values_path)?;
        return Ok(());
    }

    let values = load_score_mean_values(values_path)?;
    let current = values.len() as u64;
    if current == accepted_written {
        return Ok(());
    }

    if current > accepted_written {
        // values 超前：截断到 accepted_written
        let new_len_bytes = (accepted_written as u64 * 4) as u64;
        let f = fs_err::OpenOptions::new().write(true).open(values_path)?;
        f.set_len(new_len_bytes)?;
        return Ok(());
    }

    // values 落后：从 part 文件重建（较慢，但保证正确）
    let mut rebuilt: Vec<f32> = Vec::with_capacity(accepted_written as usize);
    for part in &manifest.parts {
        let path = output_dir.join(&part.name);
        let batch = TrainingSampleBatch::load_binary(path.to_string_lossy().as_ref())
            .with_context(|| format!("重建 score_mean_values 读取 part 失败: {}", path.display()))?;
        for s in batch.samples {
            if let Some(v) = s.value_target.first() {
                rebuilt.push(*v);
            }
        }
    }

    // 重写 values 文件
    let tmp_path = PathBuf::from(format!("{}.tmp", values_path.display()));
    let mut f = fs_err::File::create(&tmp_path)?;
    for v in &rebuilt {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush().ok();
    f.sync_all().ok();

    // Windows 替换写
    if values_path.exists() {
        fs_err::remove_file(values_path)?;
    }
    fs_err::rename(&tmp_path, values_path)?;
    Ok(())
}

// ============================================================================
// 分位数计算（基于 score_mean_values.bin 或内存 Vec）
// ============================================================================

pub fn calc_score_mean_summary(values: &mut [f32]) -> ManifestScoreMeanSummary {
    if values.is_empty() {
        return ManifestScoreMeanSummary {
            min: None,
            mean: None,
            p50: None,
            p90: None,
            p99: None,
        };
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();

    let min = values.first().cloned().unwrap_or(0.0) as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let p = |q: f64| -> f64 {
        if n == 1 {
            return values[0] as f64;
        }
        let idx = ((n - 1) as f64 * q).round() as usize;
        values[idx.min(n - 1)] as f64
    };

    ManifestScoreMeanSummary {
        min: Some(min),
        mean: Some(mean),
        p50: Some(p(0.50)),
        p90: Some(p(0.90)),
        p99: Some(p(0.99)),
    }
}
