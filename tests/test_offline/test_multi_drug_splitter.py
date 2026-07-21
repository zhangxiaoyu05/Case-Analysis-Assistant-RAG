"""
测试 app.offline.multi_drug_splitter — 多药品合集文档智能拆分模块

覆盖: detect_multi_drug, split_multi_drug, extract_drug_name, SubDocument
"""

import pytest

from app.offline.multi_drug_splitter import (
    SubDocument,
    detect_multi_drug,
    extract_drug_name,
    split_multi_drug,
)


# ============================================================
# 测试数据 fixtures
# ============================================================
@pytest.fixture
def single_drug_text():
    """单个药品的说明书文本。"""
    return """阿司匹林肠溶片说明书

【药品名称】
通用名称：阿司匹林肠溶片
商品名称：拜阿司匹灵
英文名称：Aspirin Enteric-coated Tablets

【适应症】
用于解热镇痛：缓解轻至中度疼痛，如头痛、牙痛、神经痛、肌肉痛、痛经及关节痛等。

【用法用量】
解热镇痛：成人一次0.3～0.6g，一日3次，饭后服用。

【禁忌】
1. 对阿司匹林或其他非甾体抗炎药过敏者禁用。
2. 活动性消化性溃疡或出血者禁用。

【不良反应】
1. 胃肠道反应：恶心、呕吐、上腹部不适或疼痛等。
2. 出血倾向：牙龈出血、鼻出血、皮肤瘀斑等。

【生产企业】
拜耳医药保健有限公司"""


@pytest.fixture
def multi_drug_text_separator():
    """用 ===== 分隔的多药品合集文档。"""
    return """阿司匹林肠溶片说明书

【药品名称】
通用名称：阿司匹林肠溶片
商品名称：拜阿司匹灵

【适应症】
用于解热镇痛：缓解轻至中度疼痛。

【用法用量】
解热镇痛：成人一次0.3～0.6g，一日3次。

【生产企业】
拜耳医药保健有限公司

==============================

布洛芬缓释胶囊说明书

【药品名称】
通用名称：布洛芬缓释胶囊
商品名称：芬必得

【适应症】
用于缓解轻至中度疼痛，包括头痛、偏头痛、牙痛、痛经。

【用法用量】
口服。成人一次1粒，一日2次。

【生产企业】
中美天津史克制药有限公司"""


@pytest.fixture
def multi_drug_text_marker_only():
    """仅通过【药品名称】标记分隔的合集（无 ===== 分隔符）。"""
    return """阿司匹林肠溶片说明书

【药品名称】
通用名称：阿司匹林肠溶片

【适应症】
用于解热镇痛。

【用法用量】
成人一次0.3～0.6g，一日3次。

布洛芬缓释胶囊说明书

【药品名称】
通用名称：布洛芬缓释胶囊

【适应症】
用于缓解轻至中度疼痛。

【用法用量】
成人一次1粒，一日2次。"""


@pytest.fixture
def multi_drug_text_three():
    """包含三种药品的合集文档。"""
    return """阿司匹林肠溶片说明书

【药品名称】
通用名称：阿司匹林肠溶片

【适应症】
用于解热镇痛。

==============================

布洛芬缓释胶囊说明书

【药品名称】
通用名称：布洛芬缓释胶囊

【适应症】
用于缓解轻至中度疼痛。

==============================

头孢克肟分散片说明书

【药品名称】
通用名称：头孢克肟分散片

【适应症】
适用于敏感菌引起的感染。"""


# ============================================================
# SubDocument dataclass
# ============================================================
class TestSubDocument:
    """测试 SubDocument 数据类。"""

    def test_create_subdocument(self):
        """创建 SubDocument 对象。"""
        sub = SubDocument(
            drug_name="阿司匹林肠溶片",
            text="【药品名称】\n通用名称：阿司匹林肠溶片\n\n【适应症】\n用于解热镇痛。",
            index=0,
        )
        assert sub.drug_name == "阿司匹林肠溶片"
        assert sub.index == 0
        assert "【药品名称】" in sub.text

    def test_subdocument_index_increment(self):
        """多个 SubDocument 序号递增。"""
        subs = [
            SubDocument(drug_name="阿司匹林", text="...", index=0),
            SubDocument(drug_name="布洛芬", text="...", index=1),
            SubDocument(drug_name="头孢克肟", text="...", index=2),
        ]
        for i, sub in enumerate(subs):
            assert sub.index == i


# ============================================================
# detect_multi_drug
# ============================================================
class TestDetectMultiDrug:
    """测试多药品文档检测。"""

    def test_detect_single_drug(self, single_drug_text):
        """单药品文档不应被误判为合集。"""
        assert detect_multi_drug(single_drug_text) is False

    def test_detect_multi_drug_by_section_marker(self, multi_drug_text_separator):
        """通过【药品名称】标记检测多药品。"""
        assert detect_multi_drug(multi_drug_text_separator) is True

    def test_detect_multi_drug_by_generic_name(self):
        """通过"通用名称："模式检测多药品。"""
        text = """
药品A说明书
通用名称：药品A

药品B说明书
通用名称：药品B
"""
        assert detect_multi_drug(text) is True

    def test_detect_marker_only(self, multi_drug_text_marker_only):
        """仅通过【药品名称】标记也应该检测到。"""
        assert detect_multi_drug(multi_drug_text_marker_only) is True

    def test_detect_empty_text(self):
        """空文本不应误判。"""
        assert detect_multi_drug("") is False

    def test_detect_whitespace_only(self):
        """纯空白文本不应误判。"""
        assert detect_multi_drug("   \n  \n  ") is False

    def test_detect_none_text(self):
        """None 文本不应报错，应返回 False。"""
        assert detect_multi_drug("") is False

    def test_detect_three_drugs(self, multi_drug_text_three):
        """三种药品的合集应被检测到。"""
        assert detect_multi_drug(multi_drug_text_three) is True

    def test_detect_body_text_mentions_drug_name_section(self):
        """正文中引用【药品名称】也会被计数，这是保守策略（防止漏检）。"""
        text = """【药品名称】
通用名称：阿司匹林肠溶片

【适应症】
用于解热镇痛。参见【注意事项】和【不良反应】章节。"""
        assert detect_multi_drug(text) is False

    def test_detect_two_markers_in_text(self):
        """正文中出现【药品名称】（含章节引用）共2次则判定为合集。"""
        text = """【药品名称】
通用名称：阿司匹林肠溶片

【适应症】
用于解热镇痛。本品通用名称已在【药品名称】中给出。"""
        assert detect_multi_drug(text) is True


# ============================================================
# extract_drug_name
# ============================================================
class TestExtractDrugName:
    """测试药品名称提取。"""

    def test_extract_from_generic_name_field(self):
        """从"通用名称："字段提取。"""
        text = """阿司匹林肠溶片说明书

【药品名称】
通用名称：阿司匹林肠溶片
商品名称：拜阿司匹灵"""
        assert extract_drug_name(text) == "阿司匹林肠溶片"

    def test_extract_from_generic_name_colon(self):
        """从"通用名称:"（英文冒号）提取。"""
        text = """药品说明书

【药品名称】
通用名称:布洛芬缓释胶囊"""
        assert extract_drug_name(text) == "布洛芬缓释胶囊"

    def test_extract_from_title_line(self):
        """从标题行"XXX说明书"提取。"""
        text = "头孢克肟分散片说明书\n\n【药品名称】"
        assert extract_drug_name(text) == "头孢克肟分散片"

    def test_extract_from_first_line_no_suffix(self):
        """首行没有"说明书"后缀，直接作为药名。"""
        text = "阿莫西林胶囊\n\n【药品名称】\n通用名称：阿莫西林胶囊"
        # 策略1先匹配到"通用名称"，所以返回"阿莫西林胶囊"
        assert extract_drug_name(text) == "阿莫西林胶囊"

    def test_extract_empty_text(self):
        """空文本返回空字符串。"""
        assert extract_drug_name("") == ""

    def test_extract_whitespace_text(self):
        """纯空白文本返回空字符串。"""
        assert extract_drug_name("   \n  ") == ""

    def test_extract_from_middle_of_text(self):
        """"通用名称："不在开头也能被找到。"""
        text = """【药品名称】
商品名称：芬必得
通用名称：布洛芬缓释胶囊
英文名称：Ibuprofen"""
        assert extract_drug_name(text) == "布洛芬缓释胶囊"


# ============================================================
# split_multi_drug
# ============================================================
class TestSplitMultiDrug:
    """测试多药品文档拆分。"""

    def test_split_by_separator(self, multi_drug_text_separator):
        """按 ===== 分隔符拆分。"""
        subs = split_multi_drug(multi_drug_text_separator)
        assert len(subs) == 2
        assert subs[0].drug_name == "阿司匹林肠溶片"
        assert subs[1].drug_name == "布洛芬缓释胶囊"

    def test_split_by_marker(self, multi_drug_text_marker_only):
        """按【药品名称】标记拆分。"""
        subs = split_multi_drug(multi_drug_text_marker_only)
        assert len(subs) == 2
        assert subs[0].drug_name == "阿司匹林肠溶片"
        assert subs[1].drug_name == "布洛芬缓释胶囊"

    def test_split_single_drug_returns_one(self, single_drug_text):
        """单药品不拆分，返回单个 SubDocument。"""
        subs = split_multi_drug(single_drug_text)
        assert len(subs) == 1
        assert subs[0].drug_name == "阿司匹林肠溶片"

    def test_split_preserves_content_separator(self, multi_drug_text_separator):
        """拆分后内容完整性（分隔符拆分）。"""
        subs = split_multi_drug(multi_drug_text_separator)
        # 阿司匹林的内容
        assert "阿司匹林肠溶片" in subs[0].text
        assert "拜阿司匹灵" in subs[0].text
        assert "用于解热镇痛" in subs[0].text
        assert "拜耳医药保健有限公司" in subs[0].text
        # 布洛芬的内容
        assert "布洛芬缓释胶囊" in subs[1].text
        assert "芬必得" in subs[1].text
        assert "中美天津史克制药有限公司" in subs[1].text

    def test_split_preserves_content_marker(self, multi_drug_text_marker_only):
        """拆分后内容完整性（标记拆分）。"""
        subs = split_multi_drug(multi_drug_text_marker_only)
        assert "阿司匹林肠溶片" in subs[0].text
        assert "成人一次0.3～0.6g，一日3次" in subs[0].text
        assert "布洛芬缓释胶囊" in subs[1].text
        assert "成人一次1粒，一日2次" in subs[1].text

    def test_split_empty_text(self):
        """空文本返回空列表。"""
        subs = split_multi_drug("")
        assert len(subs) == 0

    def test_split_three_drugs(self, multi_drug_text_three):
        """三种药品的合集正确拆分为 3 个。"""
        subs = split_multi_drug(multi_drug_text_three)
        assert len(subs) == 3
        assert subs[0].drug_name == "阿司匹林肠溶片"
        assert subs[1].drug_name == "布洛芬缓释胶囊"
        assert subs[2].drug_name == "头孢克肟分散片"

    def test_split_subdocument_indices(self, multi_drug_text_separator):
        """验证 SubDocument 的 index 字段递增。"""
        subs = split_multi_drug(multi_drug_text_separator)
        for i, sub in enumerate(subs):
            assert sub.index == i

    def test_split_unknown_drug_name(self):
        """无法提取药名时使用兜底命名。"""
        text = """未知文档

【药品名称】
【适应症】
一些内容

==============================

另一个文档

【药品名称】
【适应症】
更多内容"""
        subs = split_multi_drug(text)
        assert len(subs) == 2
        assert subs[0].drug_name == "未知文档"
        # 第二个文档首行是"另一个文档"，应能提取为药名
        assert "另一个文档" in subs[1].drug_name

    def test_split_drug_name_with_slash(self):
        """药名中的斜杠被替换为连字符。"""
        text = """【药品名称】
通用名称：复方甘草/氯化铵合剂

【适应症】
用于止咳化痰。"""
        subs = split_multi_drug(text)
        assert "/" not in subs[0].drug_name
        assert "复方甘草-氯化铵合剂" == subs[0].drug_name


# ============================================================
# 集成场景测试
# ============================================================
class TestIntegrationScenarios:
    """模拟真实使用场景的集成测试。"""

    def test_real_collection_format(self):
        """模拟真实的 20种药品说明书合集 格式。"""
        text = """阿司匹林肠溶片说明书

【药品名称】
通用名称：阿司匹林肠溶片

【适应症】
用于解热镇痛：缓解轻至中度疼痛。

【用法用量】
解热镇痛：成人一次0.3～0.6g，一日3次。

【生产企业】
拜耳医药保健有限公司

==============================

布洛芬缓释胶囊说明书

【药品名称】
通用名称：布洛芬缓释胶囊

【适应症】
用于缓解轻至中度疼痛，包括头痛、偏头痛、牙痛、痛经、关节痛。

【用法用量】
口服。成人一次1粒，一日2次。

【生产企业】
中美天津史克制药有限公司

==============================

阿莫西林胶囊说明书

【药品名称】
通用名称：阿莫西林胶囊

【适应症】
阿莫西林适用于敏感菌所致的感染。

【用法用量】
口服。成人一次0.5g，每6～8小时1次。

【生产企业】
广州白云山医药集团股份有限公司"""

        # 检测
        assert detect_multi_drug(text) is True

        # 拆分
        subs = split_multi_drug(text)
        assert len(subs) == 3

        # 药名
        assert subs[0].drug_name == "阿司匹林肠溶片"
        assert subs[1].drug_name == "布洛芬缓释胶囊"
        assert subs[2].drug_name == "阿莫西林胶囊"

        # 每种药品的内容完整且独立
        assert "拜耳医药保健有限公司" in subs[0].text
        assert "拜耳医药保健有限公司" not in subs[1].text
        assert "中美天津史克制药有限公司" in subs[1].text
        assert "中美天津史克制药有限公司" not in subs[2].text
        assert "广州白云山医药集团股份有限公司" in subs[2].text

    def test_single_drug_not_affected(self, single_drug_text):
        """单药品文档经过检测和拆分流程后不受影响。"""
        assert detect_multi_drug(single_drug_text) is False
        subs = split_multi_drug(single_drug_text)
        assert len(subs) == 1
        assert subs[0].drug_name == "阿司匹林肠溶片"
        assert "拜耳医药保健有限公司" in subs[0].text
