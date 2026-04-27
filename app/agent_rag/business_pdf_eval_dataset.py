from __future__ import annotations

from typing import TypedDict


class TestQuestion(TypedDict):
    question: str
    ground_truth: str


SOURCE_FILE = "деловой китайский язык.pdf"

TEST_QUESTIONS: list[TestQuestion] = [
    {
        "question": "Как по-китайски визитная карточка?",
        "ground_truth": "Визитная карточка по-китайски — 名片, pinyin: míngpiàn.",
    },
    {
        "question": "Как по-китайски директор?",
        "ground_truth": "Директор по-китайски — 经理, pinyin: jīnglǐ.",
    },
    {
        "question": "Как по-китайски компания?",
        "ground_truth": "Компания по-китайски — 公司, pinyin: gōngsī.",
    },
    {
        "question": "Как по-китайски отдел кадров или трудовые ресурсы?",
        "ground_truth": "Трудовые ресурсы по-китайски — 人力资源, pinyin: rénlì zīyuán.",
    },
    {
        "question": "Как по-китайски финансовый отдел?",
        "ground_truth": "Финансы — 财务, pinyin: cáiwù; отдел — 部, pinyin: bù.",
    },
    {
        "question": "Как по-китайски резюме?",
        "ground_truth": "Резюме по-китайски — 简历, pinyin: jiǎnlì.",
    },
    {
        "question": "Как по-китайски опыт?",
        "ground_truth": "Опыт по-китайски — 经验, pinyin: jīngyàn.",
    },
    {
        "question": "Как по-китайски должность?",
        "ground_truth": "Должность по-китайски — 职位, pinyin: zhíwèi.",
    },
    {
        "question": "Как по-китайски принимать на работу?",
        "ground_truth": "Принимать на работу по-китайски — 录用, pinyin: lùyòng.",
    },
    {
        "question": "Как по-китайски заявление или подавать заявление?",
        "ground_truth": "Подавать заявление по-китайски — 申请, pinyin: shēnqǐng.",
    },
    {
        "question": "Как по-китайски электронная почта?",
        "ground_truth": "Электронная почта по-китайски — 电子邮件, pinyin: diànzǐ yóujiàn.",
    },
    {
        "question": "Как по-китайски расписание?",
        "ground_truth": "Расписание по-китайски — 日程, pinyin: rìchéng.",
    },
    {
        "question": "Как по-китайски офис?",
        "ground_truth": "Офис по-китайски — 办公室, pinyin: bàngōngshì.",
    },
    {
        "question": "Как по-китайски бронировать?",
        "ground_truth": "Бронировать по-китайски — 预定, pinyin: yùdìng.",
    },
    {
        "question": "Как по-китайски авиабилет?",
        "ground_truth": "Авиабилет по-китайски — 机票, pinyin: jīpiào.",
    },
    {
        "question": "Как по-китайски доставка товара?",
        "ground_truth": "Доставка товара по-китайски — 送货, pinyin: sònghuò.",
    },
    {
        "question": "Как по-китайски мебель?",
        "ground_truth": "Мебель по-китайски — 家具, pinyin: jiājù.",
    },
    {
        "question": "Как по-китайски лифт?",
        "ground_truth": "Лифт по-китайски — 电梯, pinyin: diàntī.",
    },
    {
        "question": "Как по-китайски этаж?",
        "ground_truth": "Этаж по-китайски — 层, pinyin: céng.",
    },
    {
        "question": "Как по-китайски принтер?",
        "ground_truth": "Принтер по-китайски — 打印机, pinyin: dǎyìnjī.",
    },
    {
        "question": "Как по-китайски ноутбук?",
        "ground_truth": "Ноутбук по-китайски — 笔记本电脑, pinyin: bǐjìběn diànnǎo.",
    },
    {
        "question": "Как по-китайски бизнес-банкет?",
        "ground_truth": "Бизнес-банкет по-китайски — 商务宴会, pinyin: shāngwù yànhuì.",
    },
    {
        "question": "Как по-китайски сотрудничать?",
        "ground_truth": "Сотрудничать по-китайски — 合作, pinyin: hézuò.",
    },
    {
        "question": "Как по-китайски качество?",
        "ground_truth": "Качество по-китайски — 质量, pinyin: zhìliàng.",
    },
    {
        "question": "Как по-китайски цена?",
        "ground_truth": "Цена по-китайски — 价格, pinyin: jiàgé.",
    },
    {
        "question": "Как по-китайски импорт?",
        "ground_truth": "Импорт по-китайски — 进口, pinyin: jìnkǒu.",
    },
    {
        "question": "Как по-китайски экспорт?",
        "ground_truth": "Экспорт по-китайски — 出口, pinyin: chūkǒu.",
    },
    {
        "question": "Как по-китайски интернет?",
        "ground_truth": "Интернет по-китайски — 网络, pinyin: wǎngluò.",
    },
    {
        "question": "Как по-китайски совещание?",
        "ground_truth": "Совещание по-китайски — 会议, pinyin: huìyì.",
    },
    {
        "question": "Как по-китайски гарантия или гарантировать?",
        "ground_truth": "Гарантировать по-китайски — 保证, pinyin: bǎozhèng.",
    },
    {
        "question": "Как по-китайски информация?",
        "ground_truth": "Информация по-китайски — 信息, pinyin: xìnxī.",
    },
    {
        "question": "Как по-китайски загружать файл?",
        "ground_truth": "Загружать по-китайски — 上传, pinyin: shàngchuán.",
    },
    {
        "question": "Как по-китайски скачивать?",
        "ground_truth": "Скачивать по-китайски — 下载, pinyin: xiàzǎi.",
    },
    {
        "question": "Как по-китайски поисковая система?",
        "ground_truth": "Поисковая система по-китайски — 搜索引擎, pinyin: sōusuǒ yǐnqíng.",
    },
    {
        "question": "Как по-китайски бизнес-консультация?",
        "ground_truth": "Бизнес-консультация по-китайски — 商业咨询, pinyin: shāngyè zīxún.",
    },
    {
        "question": "Как по-китайски предприниматель?",
        "ground_truth": "Предприниматель по-китайски — 企业家, pinyin: qǐyèjiā.",
    },
    {
        "question": "Как по-китайски консультант?",
        "ground_truth": "Консультант по-китайски — 顾问, pinyin: gùwèn.",
    },
    {
        "question": "Как по-китайски преимущество?",
        "ground_truth": "Преимущество по-китайски — 优势, pinyin: yōushì.",
    },
    {
        "question": "Как по-китайски прибыль?",
        "ground_truth": "Прибыль по-китайски — 利润, pinyin: lìrùn.",
    },
    {
        "question": "Как по-китайски бренд или марка?",
        "ground_truth": "Бренд или марка по-китайски — 品牌, pinyin: pǐnpái.",
    },
    {
        "question": "Как по-китайски культура предприятия?",
        "ground_truth": "Культура предприятия по-китайски — 企业文化, pinyin: qǐyè wénhuà.",
    },
    {
        "question": "Как по-китайски доход?",
        "ground_truth": "Доход по-китайски — 收入, pinyin: shōurù.",
    },
    {
        "question": "Как по-китайски социальное обеспечение или福利?",
        "ground_truth": "Социальное обеспечение по-китайски — 福利, pinyin: fúlì.",
    },
    {
        "question": "Как по-китайски возможность?",
        "ground_truth": "Возможность по-китайски — 机会, pinyin: jīhuì.",
    },
    {
        "question": "Как по-китайски атмосфера?",
        "ground_truth": "Атмосфера по-китайски — 气氛, pinyin: qìfēn.",
    },
    {
        "question": "Как по-китайски кадровое агентство?",
        "ground_truth": "Кадровое агентство по-китайски — 猎头公司, pinyin: liètóu gōngsī.",
    },
    {
        "question": "Как по-китайски деловая переписка?",
        "ground_truth": "Деловая переписка по-китайски — 商务信函, pinyin: shāngwù xìnhán.",
    },
    {
        "question": "Как по-китайски письмо-запрос?",
        "ground_truth": "Письмо-запрос по-китайски — 询价, pinyin: xúnjià.",
    },
    {
        "question": "Как по-китайски предложение или оферта?",
        "ground_truth": "Предложение или оферта по-китайски — 发价, pinyin: fājià.",
    },
    {
        "question": "Как по-китайски письмо-благодарность?",
        "ground_truth": "Письмо-благодарность по-китайски — 感谢信, pinyin: gǎnxièxìn.",
    },
    {
        "question": "Как по-китайски приглашение?",
        "ground_truth": "Приглашение по-китайски — 邀请信, pinyin: yāoqǐngxìn.",
    },
]
