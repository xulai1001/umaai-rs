use std::default::Default;

use rand::{Rng, rngs::StdRng};
use serde::{Deserialize, Serialize};

//use colored::Colorize;
use crate::{explain::Explain, utils::*};

/// 继承信息，剧本通用
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InheritInfo {
    /// 蓝因子数量
    pub blue_count: Array5,
    /// 剧本因子属性
    pub extra_count: Array6
}

impl InheritInfo {
    pub fn explain(&self) -> String {
        format!(
            "种马因子: {} 剧本因子: {}",
            Explain::five_status(&self.blue_count),
            Explain::status_with_pt(&self.extra_count)
        )
    }

    /// 获取开局属性
    pub fn inherit_newgame(&self) -> Array5 {
        let mut ret = self.blue_count.clone();
        for i in 0..5 {
            ret[i] *= 7;
        }
        ret
    }

    /// 获取局中继承属性
    pub fn inherit(&self, rng: &mut StdRng) -> Array6 {
        let mut ret = Array6::default();
        let extra_scale: f64 = rng.random_range(0.0..2.0);
        for i in 0..5 {
            // 属性: 蓝因子 x 6+剧本因子 x extra_scale
            ret[i] = self.blue_count[i] * 6 + (extra_scale * self.extra_count[i] as f64).round() as i32;
        }
        // pt
        ret[5] = (self.extra_count[5] as f64 * (0.5 + 0.5 * extra_scale)).round() as i32;
        ret
    }

    /// 获取开局继承上限
    pub fn inherit_limit_newgame(&self) -> Array5 {
        let mut ret = self.blue_count.clone();
        for i in 0..5 {
            ret[i] = (ret[i] as f32 * 5.33) as i32; // 3星蓝因子提供16上限
        }
        ret
    }

    /// 获取局中继承上限 0-8
    pub fn inherit_limit(&self, _rng: &mut StdRng) -> Array5 {
        let mut ret = self.blue_count.clone();
        for i in 0..5 {
            ret[i] = (ret[i] as f32 * 5.33) as i32;
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_inherit() -> Result<()> {
        let ii = InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 40]
        };
        let mut rng = StdRng::from_os_rng();
        println!(
            "newgame: {:?} limit: {:?}",
            ii.inherit_newgame(),
            ii.inherit_limit_newgame()
        );
        println!(
            "first time: {:?} limit: {:?}",
            ii.inherit(&mut rng),
            ii.inherit_limit(&mut rng)
        );
        println!(
            "second time: {:?} limit: {:?}",
            ii.inherit(&mut rng),
            ii.inherit_limit(&mut rng)
        );
        Ok(())
    }
}
