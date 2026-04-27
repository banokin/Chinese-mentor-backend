from __future__ import annotations

from typing import TypedDict


class TestQuestion(TypedDict):
    question: str
    expected_answer: str
    expected_keywords: list[str]
    expected_page: int | None
    negative_question: bool


SOURCE_FILE = "деловой китайский язык.pdf"

EVAL_DATASET: list[TestQuestion] = [
    {
        "question": "Как по-китайски визитная карточка?",
        "expected_answer": "Визитная карточка по-китайски — 名片, pinyin: míngpiàn.",
        "expected_keywords": ["名片", "míngpiàn", "визитная карточка"],
        "expected_page": 6,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски электронная почта?",
        "expected_answer": "Электронная почта по-китайски — 电子邮件, pinyin: diànzǐ yóujiàn.",
        "expected_keywords": ["电子邮件", "diànzǐ yóujiàn", "электронная почта"],
        "expected_page": 12,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски офис?",
        "expected_answer": "Офис по-китайски — 办公室, pinyin: bàngōngshì.",
        "expected_keywords": ["办公室", "bàngōngshì", "офис"],
        "expected_page": 12,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски авиабилет?",
        "expected_answer": "Авиабилет по-китайски — 机票, pinyin: jīpiào.",
        "expected_keywords": ["机票", "jīpiào", "авиабилет"],
        "expected_page": 12,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски доставка товара?",
        "expected_answer": "Доставка товара по-китайски — 送货, pinyin: sònghuò.",
        "expected_keywords": ["送货", "sònghuò", "доставка"],
        "expected_page": 16,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски мебель?",
        "expected_answer": "Мебель по-китайски — 家具, pinyin: jiājù.",
        "expected_keywords": ["家具", "jiājù", "мебель"],
        "expected_page": 16,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски лифт?",
        "expected_answer": "Лифт по-китайски — 电梯, pinyin: diàntī.",
        "expected_keywords": ["电梯", "diàntī", "лифт"],
        "expected_page": 16,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски бизнес-банкет?",
        "expected_answer": "Бизнес-банкет по-китайски — 商务宴会, pinyin: shāngwù yànhuì.",
        "expected_keywords": ["商务宴会", "shāngwù yànhuì", "банкет"],
        "expected_page": 20,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски принимать на работу?",
        "expected_answer": "Принимать на работу по-китайски — 录用, pinyin: lùyòng.",
        "expected_keywords": ["录用", "lùyòng", "принимать на работу"],
        "expected_page": 9,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски опыт?",
        "expected_answer": "Опыт по-китайски — 经验, pinyin: jīngyàn.",
        "expected_keywords": ["经验", "jīngyàn", "опыт"],
        "expected_page": 9,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски сотрудничать?",
        "expected_answer": "Сотрудничать по-китайски — 合作, pinyin: hézuò.",
        "expected_keywords": ["合作", "hézuò", "сотрудничать"],
        "expected_page": 20,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски качество?",
        "expected_answer": "Качество по-китайски — 质量, pinyin: zhìliàng.",
        "expected_keywords": ["质量", "zhìliàng", "качество"],
        "expected_page": 20,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски цена?",
        "expected_answer": "Цена по-китайски — 价格, pinyin: jiàgé.",
        "expected_keywords": ["价格", "jiàgé", "цена"],
        "expected_page": 20,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски интернет?",
        "expected_answer": "Интернет по-китайски — 网络, pinyin: wǎngluò.",
        "expected_keywords": ["网络", "wǎngluò", "интернет"],
        "expected_page": 23,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски скачивать?",
        "expected_answer": "Скачивать по-китайски — 下载, pinyin: xiàzǎi.",
        "expected_keywords": ["下载", "xiàzǎi", "скачивать"],
        "expected_page": 26,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски предприниматель?",
        "expected_answer": "Предприниматель по-китайски — 企业家, pinyin: qǐyèjiā.",
        "expected_keywords": ["企业家", "qǐyèjiā", "предприниматель"],
        "expected_page": 27,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски консультант?",
        "expected_answer": "Консультант по-китайски — 顾问, pinyin: gùwèn.",
        "expected_keywords": ["顾问", "gùwèn", "консультант"],
        "expected_page": 27,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски преимущество?",
        "expected_answer": "Преимущество по-китайски — 优势, pinyin: yōushì.",
        "expected_keywords": ["优势", "yōushì", "преимущество"],
        "expected_page": 27,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски культура предприятия?",
        "expected_answer": "Культура предприятия по-китайски — 企业文化, pinyin: qǐyè wénhuà.",
        "expected_keywords": ["企业文化", "qǐyè wénhuà", "культура предприятия"],
        "expected_page": 30,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски деловая переписка?",
        "expected_answer": "Деловая переписка по-китайски — 商务信函, pinyin: shāngwù xìnhán.",
        "expected_keywords": ["商务信函", "shāngwù xìnhán", "деловая переписка"],
        "expected_page": 34,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски письмо-запрос?",
        "expected_answer": "Письмо-запрос по-китайски — 询价, pinyin: xúnjià.",
        "expected_keywords": ["询价", "xúnjià", "письмо-запрос"],
        "expected_page": 34,
        "negative_question": False,
    },
    {
        "question": "Как по-китайски письмо-благодарность?",
        "expected_answer": "Письмо-благодарность по-китайски — 感谢信, pinyin: gǎnxièxìn.",
        "expected_keywords": ["感谢信", "gǎnxièxìn", "письмо-благодарность"],
        "expected_page": 41,
        "negative_question": False,
    },
    {
        "question": "Как настроить Docker для FastAPI?",
        "expected_answer": "В документе нет информации для ответа на этот вопрос.",
        "expected_keywords": [],
        "expected_page": None,
        "negative_question": True,
    },
    {
        "question": "Как обучить нейросеть на PyTorch?",
        "expected_answer": "В документе нет информации для ответа на этот вопрос.",
        "expected_keywords": [],
        "expected_page": None,
        "negative_question": True,
    },
    {
        "question": "Как подключить Qdrant Cloud?",
        "expected_answer": "В документе нет информации для ответа на этот вопрос.",
        "expected_keywords": [],
        "expected_page": None,
        "negative_question": True,
    },
]

# Backward-compatible alias for older eval runner imports.
TEST_QUESTIONS = EVAL_DATASET
