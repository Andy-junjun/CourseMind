from src.schemas import Chunk
from src.study_tools import build_cloze_quiz, generate_quiz


def make_chunk(chunk_id: str, text: str, concepts: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        file_name="course.md",
        page=1,
        text=text,
        concepts=concepts,
    )


def test_generate_quiz_asks_concept_understanding_questions():
    chunks = [
        make_chunk(
            "c1",
            "激活函数会引入非线性，使神经网络能够学习复杂关系。",
            ["激活函数", "神经网络"],
        ),
        make_chunk(
            "c2",
            "梯度下降通过沿负梯度方向更新参数来减小损失函数。",
            ["梯度下降", "损失函数"],
        ),
    ]

    quiz = generate_quiz(chunks, num_questions=1)

    assert len(quiz) == 1
    assert "证据片段" not in quiz[0].question
    assert "以下哪一项最符合" in quiz[0].question or "____" in quiz[0].question
    assert quiz[0].answer in quiz[0].options
    assert "来源" in quiz[0].explanation


def test_generate_quiz_can_focus_on_recommended_concept():
    chunks = [
        make_chunk(
            "c1",
            "激活函数会引入非线性，使神经网络能够学习复杂关系。",
            ["激活函数", "神经网络"],
        ),
        make_chunk(
            "c2",
            "梯度下降通过沿负梯度方向更新参数来减小损失函数。",
            ["梯度下降", "损失函数"],
        ),
    ]

    quiz = generate_quiz(chunks, num_questions=3, target_concept="梯度下降")

    assert len(quiz) == 1
    assert quiz[0].concept == "梯度下降"
    assert quiz[0].answer in quiz[0].options
    assert "梯度下降" in quiz[0].explanation


def test_generate_quiz_masks_keywords_and_does_not_always_put_answer_first():
    chunks = [
        make_chunk(
            "c1",
            "激活函数会引入非线性，使神经网络能够学习复杂关系。",
            ["激活函数", "神经网络"],
        ),
        make_chunk(
            "c2",
            "梯度下降通过沿负梯度方向更新参数来减小损失函数。",
            ["梯度下降", "损失函数"],
        ),
        make_chunk(
            "c3",
            "卷积层用于提取局部特征，池化层可以减少特征图尺寸。",
            ["卷积层", "池化层"],
        ),
    ]

    quiz = generate_quiz(chunks, num_questions=3)

    assert any("____" in item.question for item in quiz)
    assert all(not item.question.startswith("完形填空") for item in quiz)
    assert all("空格处最合适" not in item.question for item in quiz)
    assert all(item.answer in item.options for item in quiz)
    assert any(item.options.index(item.answer) != 0 for item in quiz)


def test_cloze_quiz_does_not_mask_heading_prefix():
    chunk = make_chunk(
        "c1",
        "卷积网络结构：卷积网络是由卷积层、汇聚层、全连接层交叉堆叠而成。",
        ["卷积网络结构", "卷积层", "汇聚层", "全连接层"],
    )

    item = build_cloze_quiz(chunk, "卷积网络结构", chunk.text, [chunk])

    assert item is not None
    assert not item.question.startswith("____：")
    assert "____" in item.question
