//! 测试用的最小 ONNX 模型生成器（**不进入正式构建**）
//!
//! # 为什么是一个 `#[path]` 共享文件
//!
//! `umasim` 与 `umaai` 两侧都需要「不依赖正式权重就能加载的模型」来跑行为测试
//! （加载契约、来源标注、执行分支），而 `saved_models/` 不入库。
//!
//! 本文件放在 workspace 根的 `testsupport/`，两个 crate 在 `#[cfg(test)]` 下用
//! `#[path]` 各自引入一次，不做成生产公开 API，也不各抄一份。
//! 因此它**不在任何 crate 的模块树里**、不参与正式构建、不引入任何新依赖。
//!
//! ```ignore
//! #[path = "../../../../testsupport/onnx_fixture.rs"]
//! mod onnx_fixture;
//! ```
//!
//! # 手写 protobuf 而不是依赖 onnx 库
//!
//! 只用到 ONNX protobuf 里这几个消息的最小子集，手写比引一个序列化依赖便宜：
//! `ModelProto{1:ir_version, 2:producer_name, 7:graph, 8:opset_import}`、
//! `GraphProto{1:node, 2:name, 5:initializer, 11:input, 12:output}`、
//! `NodeProto{1:input, 2:output, 3:name, 4:op_type}`、
//! `ValueInfoProto{1:name, 2:type}`、`TypeProto{1:tensor_type}`、
//! `TypeProto.Tensor{1:elem_type, 2:shape}`、`TensorShapeProto{1:dim}`、
//! `Dimension{1:dim_value}`、`TensorProto{1:dims, 2:data_type, 8:name, 9:raw_data}`、
//! `OperatorSetIdProto{1:domain, 2:version}`。
//!
//! # 覆盖面注意
//!
//! [`matmul_model`] 的权重全 0，输出恒为 0 —— 所有候选 logit 相同，argmax 永远落在
//! 下标 0。要让「两侧推荐相同 / 不同」两条分支都真的走到，用 [`const_logits_model`]
//! 显式指定输出。**全零 fixture 不能用来证明任何与「选了哪个」有关的结论。**

use anyhow::{Result, ensure};

/// 追加一个 protobuf varint
fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// 追加一个 `字段号 + wire type` 标签
fn tag(field: u32, wire: u32, out: &mut Vec<u8>) {
    varint(u64::from((field << 3) | wire), out);
}

/// 追加一个 varint 字段
fn put_varint(field: u32, v: u64, out: &mut Vec<u8>) {
    tag(field, 0, out);
    varint(v, out);
}

/// 追加一个 length-delimited 字段（字符串 / 字节串 / 嵌套消息）
fn put_bytes(field: u32, v: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(v.len() as u64, out);
    out.extend_from_slice(v);
}

/// `TypeProto`：元素类型恒为 FLOAT(1)，形状为给定的具体维度
fn type_proto(dims: &[usize]) -> Vec<u8> {
    let mut shape = Vec::new();
    for &d in dims {
        let mut dim = Vec::new();
        put_varint(1, d as u64, &mut dim);
        put_bytes(1, &dim, &mut shape);
    }
    let mut tensor = Vec::new();
    put_varint(1, 1, &mut tensor);
    put_bytes(2, &shape, &mut tensor);
    let mut ty = Vec::new();
    put_bytes(1, &tensor, &mut ty);
    ty
}

/// `ValueInfoProto`
fn value_info(name: &str, dims: &[usize]) -> Vec<u8> {
    let mut v = Vec::new();
    put_bytes(1, name.as_bytes(), &mut v);
    put_bytes(2, &type_proto(dims), &mut v);
    v
}

/// 单输入单输出的 `NodeProto`
fn node(op: &str, name: &str, input: &str, output: &str) -> Vec<u8> {
    let mut n = Vec::new();
    put_bytes(1, input.as_bytes(), &mut n);
    put_bytes(2, output.as_bytes(), &mut n);
    put_bytes(3, name.as_bytes(), &mut n);
    put_bytes(4, op.as_bytes(), &mut n);
    n
}

/// 双输入单输出的 `NodeProto`（`MatMul` / `Add` 用）
fn node2(op: &str, name: &str, a: &str, b: &str, output: &str) -> Vec<u8> {
    let mut n = Vec::new();
    put_bytes(1, a.as_bytes(), &mut n);
    put_bytes(1, b.as_bytes(), &mut n);
    put_bytes(2, output.as_bytes(), &mut n);
    put_bytes(3, name.as_bytes(), &mut n);
    put_bytes(4, op.as_bytes(), &mut n);
    n
}

/// 给定数值的 f32 `TensorProto` initializer
fn float_initializer(name: &str, dims: &[usize], data: &[f32]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(data.len() * 4);
    for v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let mut t = Vec::new();
    for &d in dims {
        put_varint(1, d as u64, &mut t);
    }
    put_varint(2, 1, &mut t);
    put_bytes(8, name.as_bytes(), &mut t);
    put_bytes(9, &raw, &mut t);
    t
}

/// 全 0 的 f32 `TensorProto` initializer
fn zero_initializer(name: &str, rows: usize, cols: usize) -> Vec<u8> {
    float_initializer(name, &[rows, cols], &vec![0.0f32; rows * cols])
}

/// 把 `GraphProto` 包成完整 `ModelProto`
fn wrap_model(graph: Vec<u8>) -> Vec<u8> {
    let mut opset = Vec::new();
    put_bytes(1, b"", &mut opset);
    put_varint(2, 13, &mut opset);
    let mut m = Vec::new();
    put_varint(1, 7, &mut m);
    put_bytes(2, b"umaai-test-fixture", &mut m);
    put_bytes(7, &graph, &mut m);
    put_bytes(8, &opset, &mut m);
    m
}

/// 输出形状 = 输入形状的图（`Identity`）：输出维度**错**的负向 fixture
#[allow(dead_code)]
pub fn identity_model(dim: usize) -> Vec<u8> {
    let mut g = Vec::new();
    put_bytes(1, &node("Identity", "n0", "X", "Y"), &mut g);
    put_bytes(2, b"identity", &mut g);
    put_bytes(11, &value_info("X", &[1, dim]), &mut g);
    put_bytes(12, &value_info("Y", &[1, dim]), &mut g);
    wrap_model(g)
}

/// 两个输出的图：输出**个数**错的负向 fixture
#[allow(dead_code)]
pub fn two_output_model(dim: usize) -> Vec<u8> {
    let mut g = Vec::new();
    put_bytes(1, &node("Identity", "n0", "X", "Y"), &mut g);
    put_bytes(1, &node("Identity", "n1", "X", "Z"), &mut g);
    put_bytes(2, b"two_outputs", &mut g);
    put_bytes(11, &value_info("X", &[1, dim]), &mut g);
    put_bytes(12, &value_info("Y", &[1, dim]), &mut g);
    put_bytes(12, &value_info("Z", &[1, dim]), &mut g);
    wrap_model(g)
}

/// `X[1,in] @ W[in,out]` 的图，W 全 0：输出契约**正确**的正向 fixture
///
/// 输出恒为全 0，所有候选 logit 相同。只适合用来验「加载 / 契约 / 分支走向」，
/// **不适合**验任何与「选了哪个候选」有关的东西——那种请用 [`const_logits_model`]。
#[allow(dead_code)]
pub fn matmul_model(input_dim: usize, output_dim: usize) -> Vec<u8> {
    let mut g = Vec::new();
    put_bytes(1, &node2("MatMul", "n0", "X", "W", "Y"), &mut g);
    put_bytes(2, b"matmul", &mut g);
    put_bytes(5, &zero_initializer("W", input_dim, output_dim), &mut g);
    put_bytes(11, &value_info("X", &[1, input_dim]), &mut g);
    put_bytes(12, &value_info("Y", &[1, output_dim]), &mut g);
    wrap_model(g)
}

/// 输出**与输入无关、恒等于给定向量**的图：`X @ W(全0) + B(logits)`
///
/// 用它把网络的偏好钉死，才能让「网络与参照推荐相同」「不同」两条分支都真的走到——
/// 全 0 fixture 下 argmax 恒为 0，关键分支会碰巧没被覆盖而测试照样绿。
///
/// # 错误
///
/// `logits` 为空时返回错误。夹具只在测试里用，但同样走 `Result`——本项目测试
/// 一律不新增 assert 宏，失败信息随调用方的 `?` 一起上抛。
#[allow(dead_code)]
pub fn const_logits_model(input_dim: usize, logits: &[f32]) -> Result<Vec<u8>> {
    ensure!(!logits.is_empty(), "const_logits_model 需要非空 logits");
    let output_dim = logits.len();
    let mut g = Vec::new();
    put_bytes(1, &node2("MatMul", "n0", "X", "W", "H"), &mut g);
    put_bytes(1, &node2("Add", "n1", "H", "B", "Y"), &mut g);
    put_bytes(2, b"const_logits", &mut g);
    put_bytes(5, &zero_initializer("W", input_dim, output_dim), &mut g);
    put_bytes(5, &float_initializer("B", &[1, output_dim], logits), &mut g);
    put_bytes(11, &value_info("X", &[1, input_dim]), &mut g);
    put_bytes(12, &value_info("Y", &[1, output_dim]), &mut g);
    Ok(wrap_model(g))
}
