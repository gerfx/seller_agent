from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain.schema import HumanMessage, SystemMessage
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V3",
#     max_new_tokens=200,
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
#     do_sample=False

# )
load_dotenv()
giga_key = os.environ.get("GIGACHAT_CREDENTIALS")

giga = GigaChat(credentials=giga_key,
                model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)
giga.verbose = False


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Конфигурация
@dataclass
class Config:
    telegram_token: str = os.getenv("TELEGRAM_API_TOKEN", "")
    max_conversation_length: int = 10
    response_timeout: int = 30
    
config = Config()

class ClientStatus(Enum):
    COLD_LEAD = "холодный лид"
    WARM_LEAD = "теплый лид"
    NEGOTIATION = "переговоры"
    CLOSING = "закрытие сделки"
    CUSTOMER = "клиент"
    CHURNED = "потерянный"

class MessageType(Enum):
    INQUIRY = "запрос"
    OBJECTION = "возражение"
    INTEREST = "проявление интереса"
    PRICE_QUESTION = "вопрос по цене"
    COMPLAINT = "жалоба"
    THANK_YOU = "благодарность"

@dataclass
class ClientProfile:
    client_id: str
    name: str
    last_purchase: Optional[str] = None
    budget: Optional[int] = None
    status: ClientStatus = ClientStatus.COLD_LEAD
    preferences: List[str] = None
    conversation_history: List[Dict] = None
    last_interaction: Optional[datetime] = None
    pain_points: List[str] = None
    decision_maker: bool = True
    urgency_level: int = 1  # 1-5
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.pain_points is None:
            self.pain_points = []

# FSM состояния
class SalesStates(StatesGroup):
    waiting_for_message = State()
    collecting_requirements = State()
    presenting_offer = State()
    handling_objections = State()

# Система промптов с использованием Chain-of-Thought и Few-Shot
class AdvancedPromptSystem:
    def __init__(self):
        self.base_persona = """
Ты — эксперт по продажам премиум автомобилей с большим опытом.
Твои ключевые навыки:
- Консультативные продажи
- Работа с возражениями  
- Построение долгосрочных отношений
- Понимание психологии покупателей

ВАЖНЫЕ ПРИНЦИПЫ:
1. Всегда ставь потребности клиента на первое место
2. Задавай уточняющие вопросы только при необходимости и по одной теме за раз
3. Используй данные CRM для персонализации ответа
4. Предлагай решения, а не просто продукты
5. Будь честным о ценах и условиях
"""

        # Улучшенный промпт для анализа, фокус на разборе без лишних вопросов
        self.message_analysis_prompt = PromptTemplate(
            input_variables=["message", "conversation_history"],
            template="""
Проанализируй сообщение клиента и верни JSON со следующими полями:
- message_type: ("inquiry", "objection", "interest", "price_question", "complaint", "thank_you")
- interest_level: число 1-5
- emotions: список ключевых эмоций
- objections: список возражений (если есть)
- next_action: кратко ("clarify_budget", "offer_alternative", "book_test_drive", "provide_info")
- urgency: число 1-5

СООБЩЕНИЕ: "{message}"
ИСТОРИЯ: {conversation_history}
"""
        )

        self.response_generation_prompt = PromptTemplate(
            input_variables=[
                "client_profile", "analysis", "conversation_context", "available_offers", "market_insights"
            ],
            template="""
{base_persona}

ПРОФИЛЬ КЛИЕНТА:
{client_profile}

АНАЛИЗ Сообщения (JSON):
{analysis}

ИСТОРИЯ РАЗГОВОРА:
{conversation_context}

ДОСТУПНЫЕ ПРЕДЛОЖЕНИЯ:
{available_offers}

РЫНОЧНЫЕ УСЛОВИЯ:
{market_insights}

Используя SPIN-продажи, сформируй ответ, соблюдая следующие правила:

1) Если не хватает ключевой информации (бюджет, предпочтения) — задай **одно** уточняющее вопрос.
2) При наличии сомнений или возражений — предложи **одно** релевантное решение или альтернативу.
3) Всегда включай призыв к следующему шагу (тест-драйв, встреча, расчёт кредитных условий).
4) Задавай максимум **один** вопрос в конце ответа, чтобы не перегружать клиента.

Ответ в естественном тоне (не более 150 слов).
""".strip())


class SalesAgent:
    def __init__(self, config: Config):
        self.config = config
        giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
        self.llm = GigaChat(credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)
        self.llm.verbose = False
        self.prompt_system = AdvancedPromptSystem()
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.conversation_memories: Dict[str, ConversationBufferWindowMemory] = {}
        
        # База знаний
        self.product_catalog = {
            "BMW X5": {
                "price_range": {
                    "min": 8_000_000,
                    "max": 12_000_000,
                    "currency": "RUB"
                },
                "features": [
                    "полный привод",
                    "премиум интерьер",
                    "экономичность"
                ],
                "target_audience": "семейные клиенты",
                "stock_count": 5,
                "available_colors": ["чёрный", "белый", "синий"]
            },
            "BMW X7": {
                "price_range": {
                    "min": 10_000_000,
                    "max": 18_000_000,
                    "currency": "RUB"
                },
                "features": [
                    "7 мест",
                    "максимальный комфорт",
                    "представительность"
                ],
                "target_audience": "VIP клиенты",
                "stock_count": 3,
                "available_colors": ["чёрный", "серебристый", "тёмно-синий"]
            },
            "Audi Q7": {
                "price_range": {
                    "min": 9_000_000,
                    "max": 15_000_000,
                    "currency": "RUB"
                },
                "features": [
                    "электрификация",
                    "продвинутые технологии",
                    "безопасность"
                ],
                "target_audience": "технологичные клиенты",
                "stock_count": 4,
                "available_colors": ["белый", "чёрный", "серый"]
            }
        }
        
        self.objection_handlers = {
            "дорого": "Понимаю ваше беспокойство по поводу цены. Давайте посчитаем стоимость владения...",
            "не определился": "Это важное решение. Какие факторы для вас наиболее критичны?",
            "нужно подумать": "Конечно, это серьезная покупка. Может, я могу ответить на конкретные вопросы?"
        }

    def get_client_profile(self, user_id: str) -> ClientProfile:
        if user_id not in self.client_profiles:
            self.client_profiles[user_id] = ClientProfile(
                client_id=user_id,
                name=f"Иван",
                last_interaction=datetime.now()
            )
        return self.client_profiles[user_id]

    def get_conversation_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        if user_id not in self.conversation_memories:
            self.conversation_memories[user_id] = ConversationBufferWindowMemory(
                k=self.config.max_conversation_length,
                return_messages=True
            )
        return self.conversation_memories[user_id]

    async def analyze_message(self, message: str, user_id: str) -> Dict[str, Any]:
        try:
            memory = self.get_conversation_memory(user_id)
            conversation_history = memory.chat_memory.messages[-5:] if memory.chat_memory.messages else []
            prompt = self.prompt_system.message_analysis_prompt.format(
                message=message,
                conversation_history=str(conversation_history)
            )
            giga_messages = [
                SystemMessage(content="Ты — эксперт по продажам премиум автомобилей. Анализируй сообщения клиента и возвращай результат в формате JSON."),
                HumanMessage(content=prompt)
            ]
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.llm.invoke(giga_messages)),
                timeout=self.config.response_timeout
            )
            try:
                analysis = json.loads(response.content.strip())
            except Exception:
                analysis = {
                    "message_type": "inquiry",
                    "interest_level": 3,
                    "emotions": ["neutral"],
                    "objections": [],
                    "next_action": "provide_information",
                    "urgency": 2
                }
            return analysis
        except asyncio.TimeoutError:
            logger.error(f"Timeout analyzing message for user {user_id}")
            return {"error": "timeout"}
        except Exception as e:
            logger.error(f"Error analyzing message: {e}")
            return {"error": str(e)}

    async def generate_response(self, message: str, user_id: str, analysis: Dict) -> str:
        """Генерация персонализированного ответа через GigaChat"""
        try:
            client_profile = self.get_client_profile(user_id)
            memory = self.get_conversation_memory(user_id)
            conversation_context = self._format_conversation_history(memory.chat_memory.messages[-3:])
            available_offers = self._get_relevant_offers(client_profile, analysis)
            market_insights = "Текущий рынок премиум автомобилей показывает рост интереса к электрификации."
            prompt = self.prompt_system.response_generation_prompt.format(
                base_persona=self.prompt_system.base_persona,
                client_profile=self._format_client_profile(client_profile),
                message=message,
                analysis=json.dumps(analysis, ensure_ascii=False, indent=2),
                conversation_context=conversation_context,
                available_offers=available_offers,
                market_insights=market_insights
            )
            giga_messages = [
                SystemMessage(content="Ты — эксперт по продажам премиум автомобилей. Отвечай на сообщения клиента максимально персонализированно."),
                HumanMessage(content=prompt)
            ]
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.llm.invoke(giga_messages)),
                timeout=self.config.response_timeout
            )
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response.content)
            self._update_client_profile(client_profile, analysis)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Извините, произошла техническая ошибка. Наш менеджер свяжется с вами в ближайшее время."

    def _format_conversation_history(self, messages: List[BaseMessage]) -> str:
        """Форматирование истории разговора"""
        if not messages:
            return "Начало разговора"
        
        formatted = []
        for msg in messages:
            role = "Клиент" if isinstance(msg, HumanMessage) else "Менеджер"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)

    def _format_client_profile(self, profile: ClientProfile) -> str:
        """Форматирование профиля клиента"""
        return f"""
Имя: {profile.name}
Статус: {profile.status.value}
Последняя покупка: {profile.last_purchase or 'Нет данных'}
Бюджет: {profile.budget or 'Не указан'}
Предпочтения: {', '.join(profile.preferences) if profile.preferences else 'Не определены'}
Болевые точки: {', '.join(profile.pain_points) if profile.pain_points else 'Не выявлены'}
Уровень срочности: {profile.urgency_level}/5
"""

    def _get_relevant_offers(self, profile: ClientProfile, analysis: Dict) -> str:
        """Получение релевантных предложений"""
        offers = []
        
        # Логика подбора предложений на основе профиля и анализа
        if profile.budget and profile.budget > 10000000:
            offers.append("BMW X7 - премиум сегмент с максимальным комфортом")
        else:
            offers.append("BMW X5 - оптимальное соотношение цены и качества")
            
        if analysis.get("urgency", 1) > 3:
            offers.append("Специальное предложение месяца - скидка до 5%")
            
        return "\n".join(offers) if offers else "Стандартная линейка автомобилей"

    def _update_client_profile(self, profile: ClientProfile, analysis: Dict):
        """Обновление профиля клиента на основе анализа"""
        profile.last_interaction = datetime.now()
        
        # Обновляем уровень срочности
        if "urgency" in analysis:
            profile.urgency_level = max(profile.urgency_level, analysis["urgency"])
        
        # Добавляем возражения как болевые точки
        if "objections" in analysis and analysis["objections"]:
            for objection in analysis["objections"]:
                if objection not in profile.pain_points:
                    profile.pain_points.append(objection)

# Удаляем все упоминания и вызовы анализа звонков, примеры звонков и функции анализа звонков

# Инициализация бота
bot = Bot(token=config.telegram_token)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
sales_agent = SalesAgent(config)

# Обработчики команд
@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.set_state(SalesStates.waiting_for_message)
    await message.answer(
        "👋 Добро пожаловать в  автосалон MillionMiles!\n\n"
        "Я ваш персональный консультант по продажам. "
        "Расскажите, что вас интересует, и я подберу идеальное решение!"
    )

@dp.message(Command("profile"))
async def cmd_profile(message: types.Message):
    profile = sales_agent.get_client_profile(str(message.from_user.id))
    profile_text = sales_agent._format_client_profile(profile)
    await message.answer(f"📋 Ваш профиль:\n{profile_text}")

@dp.message(Command("reset"))
async def cmd_reset(message: types.Message, state: FSMContext):
    user_id = str(message.from_user.id)
    if user_id in sales_agent.client_profiles:
        del sales_agent.client_profiles[user_id]
    if user_id in sales_agent.conversation_memories:
        del sales_agent.conversation_memories[user_id]
    await state.clear()
    await message.answer("🔄 Ваш профиль сброшен. Начнем сначала!")

@dp.message(F.text)
async def handle_message(message: types.Message, state: FSMContext):
    user_id = str(message.from_user.id)
    
    try:
        await bot.send_chat_action(message.chat.id, "typing")
        
        analysis = await sales_agent.analyze_message(message.text, user_id)
        
        if "error" in analysis:
            await message.answer(
                "⚠️ Произошла техническая ошибка при обработке вашего сообщения. "
                "Пожалуйста, попробуйте еще раз или обратитесь к нашему менеджеру."
            )
            return
        
        response = await sales_agent.generate_response(message.text, user_id, analysis)

        await message.answer(response)
        
        logger.info(f"User {user_id}: {message.text[:50]}... -> Response generated")
        
    except Exception as e:
        logger.error(f"Error handling message from {user_id}: {e}")
        await message.answer(
            "😔 Извините, произошла неожиданная ошибка. "
            "Наш технический специалист уже уведомлен. "
            "Попробуйте написать еще раз через несколько минут."
        )

@dp.message.middleware()
async def logging_middleware(handler, event, data):
    start_time = datetime.now()
    result = await handler(event, data)
    end_time = datetime.now()
    
    logger.info(f"Message processed in {(end_time - start_time).total_seconds():.2f}s")
    return result

async def main():
    if not config.telegram_token:
        logger.error("Missing required environment variables!")
        return
    
    logger.info("Sales Bot starting...")
    logger.info(f"Max conversation length: {config.max_conversation_length}")
    logger.info(f"Response timeout: {config.response_timeout}s")
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
    finally:
        await bot.session.close()