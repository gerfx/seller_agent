import os
import json
import logging
import asyncio
from typing import List, Literal, Dict
from pydantic import BaseModel, ValidationError

from langchain.schema import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

logger = logging.getLogger(__name__)

# Инициализация GigaChat
giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
giga = GigaChat(
    credentials=giga_key,
    model="GigaChat",
    timeout=30,
    verify_ssl_certs=False
)
giga.verbose = False

# Pydantic-модель результатов
class CallAnalysisResultModel(BaseModel):
    quality: float  # теперь float от 0 до 1
    reasons: List[str]
    recommendations: List[str]

# Разбивка звонка на этапы (схематично)
async def split_into_stages(call_text: str) -> Dict[str, str]:
    prompt = (
        "Разбей текст звонка на этапы скрипта: "
        "1) приветствие, 2) выявление потребностей, 3) презентация, "
        "4) работа с возражениями, 5) завершение. "
        "Верни JSON, где ключи — названия этапов, значения — текст."
        f"\n\nТекст:\n{call_text}"
    )
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: giga.invoke([
            SystemMessage(content="Ты — эксперт по продажам и аудиту звонков."),
            HumanMessage(content=prompt)
        ])
    )
    try:
        return json.loads(resp.content.strip())
    except Exception:
        return {}

# Шаг 1: качество звонка
async def get_quality(call_text: str) -> float:
    prompt = (
        "Проанализируй звонок и оцени его качество по шкале от 0 до 1, где 0 — очень плохой звонок, 1 — отличный звонок. "
        "Ответь только числом от 0 до 1, без пояснений."
        f"\n\nТекст:\n{call_text}"
    )
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: giga.invoke([HumanMessage(content=prompt)])
    )
    print(f"Quality response: {resp.content.strip()}")
    try:
        value = float(resp.content.strip().replace(',', '.'))
        value = max(0.0, min(1.0, value))
    except Exception:
        value = 0.0
    return value

# Шаг 2: причины
async def get_reasons(call_text: str, quality: str) -> List[str]:
    prompt = (
        f"Звонок оценён как «{quality}». Назови коротко 2–3 причины, почему ты так решил. Просто перечисли их через запятую, не используй JSON."
        f"\n\nТекст:\n{call_text}"
    )
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: giga.invoke([HumanMessage(content=prompt)])
    )
    # Разделяем по запятым и убираем лишние пробелы
    print(f"Reasons response: {resp.content.strip()}")
    return [reason.strip() for reason in resp.content.strip().split(',') if reason.strip()]

# Шаг 3: рекомендации
async def get_recommendations(call_text: str) -> List[str]:
    prompt = (
        "Дай 3 конкретных рекомендации по улучшению скрипта продаж. Просто перечисли их через запятую, не используй JSON."
        f"\n\nТекст:\n{call_text}"
    )
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: giga.invoke([HumanMessage(content=prompt)])
    )
    print(f"Recommendations response: {resp.content.strip()}")
    return [rec.strip() for rec in resp.content.strip().split(',') if rec.strip()]

# Основная функция
async def analyze_call_text(call_text: str) -> Dict:
    # 0. Эвристики

    # 1. Системное сообщение-чеклист
    giga_messages = [
        SystemMessage(content=(
            "Чек-лист хорошего звонка:\n"
            "1) Приветствие с представлением\n"
            "2) Выявление потребностей\n"
            "3) Презентация продукта/услуги\n"
            "4) Работа с возражениями\n"
            "5) Чёткое завершение и следующий шаг\n"
            "6) Вежливое общение и заинтересованность продавца\n"
        ))
    ]

    # 2. Разбивка на этапы (для консервации контекста)
    stages = await split_into_stages(call_text)
    if stages:
        giga_messages.append(
            SystemMessage(content=f"Этапы звонка и тексты:\n{json.dumps(stages, ensure_ascii=False, indent=2)}")
        )

    # 3. Chain-of-Thought: качество → причины → рекомендации
    quality = await get_quality(call_text)
    reasons = await get_reasons(call_text, quality)
    recommendations = await get_recommendations(call_text)

    # 4. Собираем финальный результат
    result_dict = {
        "quality": quality,
        "reasons": reasons ,  # добавляем эвристики в причины
        "recommendations": recommendations
    }

    # 5. Валидация Pydantic
    try:
        result = CallAnalysisResultModel(**result_dict).dict()
    except ValidationError as e:
        logger.error("Pydantic validation failed: %s", e)
        # Фолбэк
        result = {
            "quality": "bad",
            "reasons": ["Ошибка валидации результата"],
            "recommendations": ["Проверьте корректность формата JSON"]
        }

    return result

# Пример теста
if __name__ == "__main__":
    async def main():
        bad_call = """
    Менеджер: Алло.  
    Клиент: Здравствуйте, интересует BMW X5.  
    Менеджер: На сайте всё написано, посмотрите.  
    Клиент: А можно узнать, есть ли в наличии?  
    Менеджер: Не знаю, сейчас не могу посмотреть. Позвоните позже.  
    Клиент: Понял, спасибо…  
    Менеджер: Ага.
    Клиент: До свидания.
    Менеджер: До свидания.
    """
        second_bad_call = """
        Менеджер: Добрый день, слушаю вас.  
        Клиент: Здравствуйте, подыскиваю автомобиль, смотрю в сторону BMW X5.  
        Менеджер: Угу, X5 популярная модель.  
        Клиент: А какие комплектации у вас есть?  
        Менеджер: Разные, зависит от наличия.  
        Клиент: А с чем именно есть сейчас?  
        Менеджер: Сейчас не подскажу, база иногда обновляется, лучше подъехать.  
        Клиент: Я пока только смотрю.  
        Менеджер: Понятно. Если что — звоните.
        """
        good_call = """
        Менеджер: Добрый день, меня зовут Алексей, автосалон «Премиум Драйв». Подскажите, пожалуйста, вы подбираете автомобиль для себя или для кого-то другого?  
        Клиент: Для себя, думаю о BMW X5.  
        Менеджер: Отличный выбор — X5 отлично сочетает комфорт и безопасность. Скажите, вам важнее простор в салоне или экономичность?  
        Клиент: Скорее простор, у меня семья.  
        Менеджер: Понимаю, тогда могу предложить комплектацию с панорамной крышей и расширенным багажником. Сейчас в наличии есть несколько вариантов в чёрном и синем цветах. Хотите запланируем тест-драйв или обсудим кредитные условия?  
        Клиент: Да, давайте тест-драйв.
        Менеджер: Отлично, когда вам удобно? У нас есть свободные слоты на завтра и послезавтра.  
        Клиент: Завтра в 11 подойдёт.
        Менеджер: Записал вас на завтра в 11:00. Приходите, будем рады помочь с выбором! Если будут вопросы, звоните в любое время.
        клиент: Спасибо, до завтра!
        """
        sample = good_call
        res = await analyze_call_text(sample)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        logger.info("Анализ звонка завершён успешно.")

        print("=" * 50)

        sample_bad = bad_call
        res_bad = await analyze_call_text(sample_bad)
        print(json.dumps(res_bad, ensure_ascii=False, indent=2))
        logger.info("Анализ звонка завершён успешно.")
        print("=" * 50)
        sample_second_bad = second_bad_call
        res_second_bad = await analyze_call_text(sample_second_bad)
        print(json.dumps(res_second_bad, ensure_ascii=False, indent=2))
        logger.info("Анализ звонка завершён успешно.")

    asyncio.run(main())
