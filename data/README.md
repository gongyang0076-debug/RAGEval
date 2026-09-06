# 电商客服数据 v1

本目录内容为人工编写的虚构“星桥商城”客服政策与评测标注，不代表任何真实商家、
法律承诺或实时业务数据。全部文件使用 UTF-8 编码，顶层为 JSON 数组。

## Corpus

`corpus/ecommerce_v1.json` 包含 36 个 chunk，分属 6 个文档，每个文档 6 个 chunk：

| 主题 | doc_id | chunk 数 |
| --- | --- | --- |
| 订单 | order_policy | 6 |
| 支付 | payment_policy | 6 |
| 退款 | refund_policy | 6 |
| 物流 | logistics_policy | 6 |
| 优惠券 | coupon_policy | 6 |
| 会员与售后 | membership_policy | 6 |

`chunk_id` 为文档 ID 加三位序号，全局唯一。`metadata` 记录主题、中文主题名、
来源、版本与虚构标识。内容有意保留以下近似主题，帮助后续验证检索区分能力：

- 未支付取消、已支付取消、配货后拦截。
- 支付异常等待10分钟、重复扣款核实后1个工作日发起退款、一般退款48小时内发起。
- 退款发起与支付渠道3至7个工作日到账。
- 无理由退货7天、审核后寄出7天、质量售后15天。
- 商品券门槛使用优惠前金额，基础运费门槛使用优惠后金额。
- 未支付订单释放券、整单退款返券、部分退款不返券。
- 基础运费减免与偏远地区附加费不减免。

## Evaluation Dataset

`datasets/ecommerce_eval_v1.json` 共 60 条：normal 30、unanswerable 10、
invariance 12、adversarial 8。所有问题及标准答案均为人工编写，不通过模型生成。

`relevant_doc_ids` 保留原字段名，但从 Sprint 2 起明确引用 **chunk_id**，
不是 doc_id。可回答问题标注足以支撑标准答案的 chunk；跨政策的定制商品对抗题
标注两个 chunk。36 个 chunk 均至少被一条可回答 case 引用。

normal 每个主题 5 条，覆盖明确规则、边界条件和相似概念区分。
unanswerable 的 10 条均为 `answerable=false`、`relevant_doc_ids=[]`，涉及知识库
未定义的价格、电话、积分、保修年限、活动、赔偿和个人实时数据；标准答案说明
缺少哪些信息，不猜测具体数值。拒答可以利用知识库对信息边界的描述，但不将这种
边界说明标成可回答原问题的证据。

invariance 为 6 组，每组 2 个正式/口语同义问法，组内标准答案和证据一致：

| variant_group | 场景 |
| --- | --- |
| invoice_request | 企业电子发票申请 |
| duplicate_payment | 重复扣款退款 |
| refund_coupon_return | 整单退款返券 |
| damaged_parcel | 收货破损处理 |
| unpaid_coupon_release | 未支付取消释放券 |
| membership_renewal | 会员续费 |

adversarial 中 6 条有证据纠正错误前提，标记 `answerable=true`；另 2 条要求编造
隐藏优惠码或确认未经证实的会员价格，标记 `answerable=false` 且证据为空。
`answerable` 表示知识库能否给出有依据的实质答复，不表示应当服从问题中的诱导指令。

## 校验边界

Loader 校验文件结构、ID 唯一性、分块引用及不可回答约束；真实数据的测试还固定
验证主题/分类数量、分组一致性与 chunk 标注覆盖。事实一致性依赖对政策和标准答案
的人工核对，结构测试不能证明语义正确性。这是小型合成评测集，尚未经真实用户
问题分布校准，后续扩展时应复核政策版本、相关性标签和难度分布。
