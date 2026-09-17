# 文件作用：测试知识库和试卷的读取、编号、结构、标注关系和正式数据规模。
# 为什么有它：防止错误测试数据让后续指标失去意义。
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.dataset import (
    CorpusDocument,
    DataValidationError,
    EvalCase,
    load_corpus,
    load_dataset,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# 做什么：提供一条合法知识块字典。
# 为什么需要：作为加载校验测试的基础资料。
@pytest.fixture
def corpus_record():
    return dict(doc_id="refund_policy", chunk_id="refund_policy_001", content="Refund policy", metadata={})


# 做什么：提供引用该知识块的合法题目字典。
# 为什么需要：作为正确引用关系的起点。
@pytest.fixture
def case_record():
    return dict(
        id="case-1", query="How do refunds work?", expected_answer="Refund policy",
        relevant_doc_ids=["refund_policy_001"], answerable=True, category="normal",
    )


# 做什么：将基础知识块转成模型对象列表。
# 为什么需要：供 Dataset Loader 检查标注引用。
@pytest.fixture
def corpus(corpus_record):
    return [CorpusDocument(**corpus_record)]


# 做什么：把指定记录写入独立临时 JSON 文件。
# 为什么需要：测试文件读取路径而不触碰正式数据。
def write_json(tmp_path, records):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


# 做什么：验证合法知识库与试卷可以关联加载。
# 为什么需要：确认正常数据路径可用。
def test_load_corpus_and_dataset(tmp_path, corpus_record, case_record):
    corpus = load_corpus(write_json(tmp_path, [corpus_record]))
    cases = load_dataset(write_json(tmp_path, [case_record]), corpus)
    assert isinstance(corpus[0], CorpusDocument)
    assert isinstance(cases[0], EvalCase)
    assert cases[0].relevant_doc_ids == [corpus[0].chunk_id]
    assert cases[0].variant_group is None


# 做什么：验证坏 JSON 的报错包含文件和位置。
# 为什么需要：便于直接修复具体数据错误。
@pytest.mark.parametrize("kind", ["corpus", "dataset"])
def test_invalid_json_has_file_and_location(tmp_path, corpus, kind):
    path = tmp_path / "broken.json"
    path.write_text('[{"id":}]', encoding="utf-8")
    with pytest.raises(DataValidationError, match=r"broken.json: invalid JSON at line 1, column"):
        load_corpus(path) if kind == "corpus" else load_dataset(path, corpus)


# 做什么：验证顶层或字段结构错误被统一报告。
# 为什么需要：不合法记录不能进入业务流程。
@pytest.mark.parametrize("kind", ["corpus", "dataset"])
@pytest.mark.parametrize("records", [{"wrong": "root"}, [{}]])
def test_pydantic_structure_errors(tmp_path, corpus, kind, records):
    path = write_json(tmp_path, records)
    with pytest.raises(DataValidationError, match=r"invalid (CorpusDocument|EvalCase) structure"):
        load_corpus(path) if kind == "corpus" else load_dataset(path, corpus)


# 做什么：验证不同文档也不能使用相同 chunk_id。
# 为什么需要：检索命中要求知识块编号全局唯一。
def test_duplicate_chunk_ids_across_documents(tmp_path, corpus_record):
    second = {**corpus_record, "doc_id": "other_policy"}
    with pytest.raises(DataValidationError, match="duplicate chunk_id 'refund_policy_001'"):
        load_corpus(write_json(tmp_path, [corpus_record, second]))


# 做什么：验证同文档的不同知识块可以共存。
# 为什么需要：文档编号重复不应被误当成知识块重复。
def test_same_doc_id_with_distinct_chunks_is_valid(tmp_path, corpus_record):
    second = {**corpus_record, "chunk_id": "refund_policy_002"}
    assert len(load_corpus(write_json(tmp_path, [corpus_record, second]))) == 2


# 做什么：验证去掉首尾空白后的题号不能重复。
# 为什么需要：避免看起来不同却实际相同的用例编号。
def test_duplicate_case_ids_after_normalization(tmp_path, corpus, case_record):
    second = {**case_record, "id": " case-1 "}
    with pytest.raises(DataValidationError, match="case 'case-1': duplicate case id"):
        load_dataset(write_json(tmp_path, [case_record, second]), corpus)


# 做什么：验证标注只能引用真实存在的知识块编号。
# 为什么需要：不能拿文档名或不存在的块作为 ground truth。
@pytest.mark.parametrize("reference", ["missing_chunk", "refund_policy"])
def test_references_must_be_existing_chunk_ids(tmp_path, corpus, case_record, reference):
    case_record["relevant_doc_ids"] = [reference]
    with pytest.raises(DataValidationError, match=rf"case 'case-1': unknown relevant_doc_ids.*{reference}"):
        load_dataset(write_json(tmp_path, [case_record]), corpus)


# 做什么：验证可回答题必须提供相关块。
# 为什么需要：没有依据无法定义普通检索召回率。
def test_answerable_requires_evidence(tmp_path, corpus, case_record):
    case_record["relevant_doc_ids"] = []
    with pytest.raises(DataValidationError, match="answerable=true requires relevant_doc_ids"):
        load_dataset(write_json(tmp_path, [case_record]), corpus)


# 做什么：验证不可回答题不能携带相关块列表。
# 为什么需要：防止可回答性与标注相互矛盾。
@pytest.mark.parametrize("category", ["unanswerable", "adversarial"])
def test_unanswerable_flag_forbids_references(tmp_path, corpus, case_record, category):
    case_record.update(answerable=False, category=category)
    with pytest.raises(DataValidationError, match=r"answerable=false requires relevant_doc_ids=\[\]"):
        load_dataset(write_json(tmp_path, [case_record]), corpus)


# 做什么：验证不可回答类别必须标为不可回答。
# 为什么需要：避免类别标签与计分范围冲突。
def test_unanswerable_category_requires_false_flag(tmp_path, corpus, case_record):
    case_record["category"] = "unanswerable"
    with pytest.raises(DataValidationError, match="unanswerable category requires answerable=false"):
        load_dataset(write_json(tmp_path, [case_record]), corpus)


# 做什么：验证合法不可回答题可以正常加载。
# 为什么需要：安全评测需要保留这些题而非删除。
def test_valid_unanswerable_case(tmp_path, corpus, case_record):
    case_record.update(answerable=False, category="unanswerable", relevant_doc_ids=[], expected_answer="Unknown")
    cases = load_dataset(write_json(tmp_path, [case_record]), corpus)
    assert cases[0].answerable is False
    assert cases[0].relevant_doc_ids == []


# 做什么：验证同义组编号正确读取。
# 为什么需要：后续可以按组核对语义不变的样例。
def test_variant_group_parses(tmp_path, corpus, case_record):
    case_record.update(category="invariance", variant_group="refund_paraphrases")
    cases = load_dataset(write_json(tmp_path, [case_record]), corpus)
    assert cases[0].variant_group == "refund_paraphrases"


# 做什么：验证加载试卷时重新检查外部传入语料的唯一性。
# 为什么需要：调用者可能绕过 load_corpus 直接提供错误资料。
def test_dataset_rechecks_supplied_corpus_uniqueness(tmp_path, corpus, case_record):
    with pytest.raises(DataValidationError, match="supplied corpus: duplicate chunk_id"):
        load_dataset(write_json(tmp_path, [case_record]), corpus + corpus)


# 做什么：验证不存在的文件给出明确加载错误。
# 为什么需要：帮助使用者区分路径问题和数据内容问题。
@pytest.mark.parametrize("kind", ["corpus", "dataset"])
def test_missing_file_is_clear(tmp_path, corpus, kind):
    path = tmp_path / "missing.json"
    with pytest.raises(DataValidationError, match=r"missing.json: cannot read UTF-8 data"):
        load_corpus(path) if kind == "corpus" else load_dataset(path, corpus)


# 做什么：验证不能按 UTF-8 读取的文件明确报错。
# 为什么需要：避免乱码数据混入模型输入。
def test_invalid_utf8_is_clear(tmp_path):
    path = tmp_path / "bad_encoding.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(DataValidationError, match="cannot read UTF-8 data"):
        load_corpus(path)


# 做什么：验证带 UTF-8 字节顺序标记的文件也能读取。
# 为什么需要：兼容常见编辑器保存方式。
def test_utf8_bom_supported(tmp_path, corpus_record):
    path = tmp_path / "bom.json"
    path.write_text(json.dumps([corpus_record]), encoding="utf-8-sig")
    assert load_corpus(path)[0].doc_id == "refund_policy"


# 做什么：验证正式语料规模、类别分布、标注和同义组一致性。
# 为什么需要：防止示例数据被无意改坏却没有发现。
def test_demo_data_counts_labels_and_variant_consistency():
    corpus = load_corpus(DATA_DIR / "corpus" / "ecommerce_v1.json")
    cases = load_dataset(DATA_DIR / "datasets" / "ecommerce_eval_v1.json", corpus)
    assert len(corpus) == 36
    assert Counter(doc.metadata["topic"] for doc in corpus) == {
        "order": 6, "payment": 6, "refund": 6, "logistics": 6, "coupon": 6, "membership": 6,
    }
    assert len(cases) == 60
    assert Counter(case.category for case in cases) == {
        "normal": 30, "unanswerable": 10, "invariance": 12, "adversarial": 8,
    }
    assert all(case.expected_answer.strip() for case in cases)
    assert all("星桥" in doc.metadata["source"] for doc in corpus)
    assert {ref for case in cases for ref in case.relevant_doc_ids} == {doc.chunk_id for doc in corpus}
    groups = defaultdict(list)
    for case in cases:
        if case.category == "invariance":
            assert case.variant_group
            groups[case.variant_group].append(case)
        else:
            assert case.variant_group is None
    assert len(groups) == 6
    for members in groups.values():
        assert len(members) == 2
        assert len({case.query for case in members}) == 2
        assert len({case.expected_answer for case in members}) == 1
        assert len({tuple(case.relevant_doc_ids) for case in members}) == 1
        assert all(case.answerable for case in members)
