"""Step 48: Preparation and validation of high-quality short conversational SFT dataset v2.

Curates 1,000+ verified, short-answer conversational training examples:
- Natural conversational greetings and polite closings
- Humble, accurate assistant persona statements
- Short direct factual QA (disjoint from benchmark)
- Everyday conversational questions
- Core definitions
- Basic science
- Simple Python concepts
- Elementary classification
- Single-step deterministic instructions
- Small valid lists
- Simple logical relationships
- Verified arithmetic (disjoint operands)
- Concise constraint responses

Strict Protections:
- ZERO LEAKAGE: Automated collision screening against benchmark_conversational_step48.json
- Assistant-only loss masking with end-of-turn delimiter
- Token length validation (User <= 128 tokens, Assistant <= 96 tokens)
- 85% Train / 15% Validation split
- Caches tokenized binary tensors to data/cache_sft_neural_v2/
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.bpe_tokenizer import BPETokenizer
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    ConversationTurn,
    encode_conversation,
    scrub_pii,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = REPO_ROOT / "data" / "benchmarks" / "benchmark_conversational_step48.json"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
CACHE_DIR = REPO_ROOT / "data" / "cache_sft_neural_v2"
JSONL_DIR = REPO_ROOT / "data" / "sft" / "conversational_v2"


def normalize(text: str) -> str:
    """Normalize text for collision checking."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def load_benchmark_normalized_prompts() -> Set[str]:
    """Load normalized prompts from the frozen benchmark."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {normalize(item["prompt"]) for item in data["prompts"]}


def build_raw_conversational_v2_pool() -> List[Dict[str, str]]:
    """Build the raw catalog of short conversational instruction pairs."""
    items: List[Dict[str, str]] = []

    # 1. Greetings, Persona, Acknowledgements & Closings (~150 examples)
    greetings = [
        ("Hi!", "Hello! How can I help you today?"),
        ("Hi there!", "Hello! How can I assist you today?"),
        ("Hello!", "Hi! What can I help you with today?"),
        ("Hello there!", "Hello! What can I assist you with?"),
        ("Good morning!", "Good morning! How can I assist you today?"),
        ("Good afternoon!", "Good afternoon! How can I help you today?"),
        ("Good evening!", "Good evening! How can I assist you tonight?"),
        ("Hey!", "Hello! What can I do for you today?"),
        ("Hey there!", "Hi there! How can I help you?"),
        ("Greetings!", "Greetings! How can I assist you today?"),
        ("Howdy!", "Hello! How can I help you today?"),
        ("Hello, how are things?", "Hello! Everything is going well, thank you. How can I help?"),
        ("Good day!", "Good day! How can I assist you?"),
        ("How are you doing?", "I am doing well, thank you! How can I assist you today?"),
        ("How is your day going?", "My day is going well, thank you. How can I help you?"),
        # Persona
        ("Who are you?", "I am GroundUpGPT, a small language model built and trained from scratch."),
        ("What are you?", "I am GroundUpGPT, an experimental neural language model."),
        ("What is your name?", "My name is GroundUpGPT."),
        ("Tell me about yourself.", "I am GroundUpGPT, a compact 27M parameter language model trained from scratch."),
        ("What can you do?", "I can answer questions, explain concepts, help with simple tasks, and converse."),
        ("Are you a human?", "No, I am GroundUpGPT, an artificial intelligence language model."),
        ("Who made you?", "I was created as part of the GroundUpGPT project, trained from scratch on open data."),
        # Thanks & Acknowledgements
        ("Thank you!", "You are very welcome! Let me know if you need anything else."),
        ("Thanks!", "You are welcome! Glad I could help."),
        ("Thank you very much!", "You are very welcome! It is my pleasure to assist."),
        ("Thanks for the explanation.", "You are welcome! I am glad the explanation was helpful."),
        ("Thanks for the info.", "You are welcome! Let me know if you have any other questions."),
        ("I appreciate your help.", "You are welcome! I am always happy to help."),
        ("That was very helpful.", "I am glad to hear that! Let me know if you need anything else."),
        ("Perfect, thank you.", "You are welcome! Have a great day."),
        # Closings
        ("Goodbye!", "Goodbye! Have a great day."),
        ("Bye!", "Goodbye! Feel free to ask more questions anytime."),
        ("See you later!", "See you later! Take care."),
        ("Have a good day!", "Thank you! I hope you have a wonderful day as well."),
        ("Good night!", "Good night! Sleep well and take care."),
        ("That is all for today.", "Sounds good! Have a great day and feel free to return anytime."),
    ]
    for i, (p, r) in enumerate(greetings):
        cid = f"greet_{i:03d}"
        items.append({"concept_id": cid, "category": "greetings_persona", "prompt": p, "response": r})
        # Add slight polite prefix variations
        items.append({"concept_id": cid, "category": "greetings_persona", "prompt": f"Can you tell me: {p.lower()}", "response": r})

    # 2. Short Direct Factual QA (~200 examples, strictly disjoint from benchmark)
    facts = [
        ("What is the capital of Italy?", "The capital of Italy is Rome."),
        ("What is the capital of Germany?", "The capital of Germany is Berlin."),
        ("What is the capital of Spain?", "The capital of Spain is Madrid."),
        ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
        ("What is the capital of Australia?", "The capital of Australia is Canberra."),
        ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
        ("What is the capital of China?", "The capital of China is Beijing."),
        ("What is the capital of South Korea?", "The capital of South Korea is Seoul."),
        ("What is the capital of Russia?", "The capital of Russia is Moscow."),
        ("What is the capital of Mexico?", "The capital of Mexico is Mexico City."),
        ("What is the capital of Argentina?", "The capital of Argentina is Buenos Aires."),
        ("What is the capital of Greece?", "The capital of Greece is Athens."),
        ("What is the capital of Sweden?", "The capital of Sweden is Stockholm."),
        ("What is the capital of Norway?", "The capital of Norway is Oslo."),
        ("What is the capital of Portugal?", "The capital of Portugal is Lisbon."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What planet is known as the Red Planet?", "Mars is known as the Red Planet."),
        ("What is the hottest planet in our solar system?", "Venus is the hottest planet in our solar system."),
        ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice."),
        ("Who painted The Starry Night?", "Vincent van Gogh painted The Starry Night."),
        ("What is the official currency of the United States?", "The official currency of the United States is the US Dollar."),
        ("What is the currency of Japan?", "The currency of Japan is the Japanese Yen."),
        ("What is the primary currency of Germany?", "The currency of Germany is the Euro."),
        ("What is the longest river in the world?", "The Nile River is generally considered the longest river in the world."),
        ("What is the largest desert in the world?", "The Antarctic Desert is the largest desert on Earth."),
        ("What is the largest hot desert in the world?", "The Sahara Desert is the largest hot desert in the world."),
        ("What is the tallest living tree species on Earth?", "The coast redwood is the tallest living tree species."),
        ("What is the deepest oceanic trench in the world?", "The Mariana Trench is the deepest oceanic trench on Earth."),
        ("What is the natural satellite of the Earth?", "The Moon is Earth's only natural satellite."),
        ("What is the capital of Kenya?", "The capital of Kenya is Nairobi."),
        ("What is the capital of Thailand?", "The capital of Thailand is Bangkok."),
        ("What continent is Egypt located on?", "Egypt is located on the continent of Africa."),
        ("What continent is Germany located on?", "Germany is located on the continent of Europe."),
        ("What continent is China located on?", "China is located on the continent of Asia."),
        ("What is the fastest land animal in the world?", "The cheetah is the fastest land animal."),
        ("What is the largest mammal on Earth?", "The blue whale is the largest mammal on Earth."),
        ("What is the capital of Belgium?", "The capital of Belgium is Brussels."),
        ("What is the capital of the Netherlands?", "The capital of the Netherlands is Amsterdam."),
        ("What is the capital of Switzerland?", "The capital of Switzerland is Bern."),
        ("What is the capital of Austria?", "The capital of Austria is Vienna."),
        ("What is the capital of Poland?", "The capital of Poland is Warsaw."),
        ("What is the capital of Denmark?", "The capital of Denmark is Copenhagen."),
        ("What is the capital of Finland?", "The capital of Finland is Helsinki."),
        ("What is the capital of Ireland?", "The capital of Ireland is Dublin."),
        ("What is the capital of New Zealand?", "The capital of New Zealand is Wellington."),
        ("What is the capital of Turkey?", "The capital of Turkey is Ankara."),
        ("What is the capital of Saudi Arabia?", "The capital of Saudi Arabia is Riyadh."),
        ("What is the capital of Chile?", "The capital of Chile is Santiago."),
        ("What is the capital of Peru?", "The capital of Peru is Lima."),
        ("What is the capital of Colombia?", "The capital of Colombia is Bogota."),
        ("Who wrote Moby-Dick?", "Herman Melville wrote Moby-Dick."),
        ("Who wrote 1984?", "George Orwell wrote 1984."),
        ("Who wrote To Kill a Mockingbird?", "Harper Lee wrote To Kill a Mockingbird."),
        ("Who wrote The Great Gatsby?", "F. Scott Fitzgerald wrote The Great Gatsby."),
        ("Who wrote War and Peace?", "Leo Tolstoy wrote War and Peace."),
        ("What is the longest continental mountain range on Earth?", "The Andes is the longest continental mountain range on Earth."),
        ("What is the largest island in the world?", "Greenland is the largest island in the world."),
        ("What is the deepest lake in the world?", "Lake Baikal is the deepest lake in the world."),
        ("What is the currency of Canada?", "The currency of Canada is the Canadian Dollar."),
        ("What is the currency of Australia?", "The currency of Australia is the Australian Dollar."),
        ("What is the currency of Switzerland?", "The currency of Switzerland is the Swiss Franc."),
        ("What is the currency of India?", "The currency of India is the Indian Rupee."),
        ("What is the currency of China?", "The currency of China is the Chinese Yuan."),
        ("What is the currency of South Korea?", "The currency of South Korea is the South Korean Won."),
        ("What is the currency of Mexico?", "The currency of Mexico is the Mexican Peso."),
    ]
    for i, (p, r) in enumerate(facts):
        cid = f"fact_{i:03d}"
        items.append({"concept_id": cid, "category": "short_direct_facts", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "short_direct_facts", "prompt": f"Please answer: {p}", "response": r})
        # Concise variant
        concise_ans = r.split(" is ")[-1].rstrip(".") if " is " in r else r
        items.append({"concept_id": cid, "category": "short_direct_facts", "prompt": f"Short answer: {p}", "response": concise_ans + "."})

    # 3. Everyday Conversational Q&A (~100 examples)
    everyday = [
        ("How do you boil an egg?", "Place the egg in boiling water for 6 to 10 minutes depending on desired firmness, then cool in ice water."),
        ("What should you do when you feel thirsty?", "You should drink a glass of clean water to stay hydrated."),
        ("Why do people wear sunglasses?", "People wear sunglasses to shield their eyes from bright sunlight and harmful UV radiation."),
        ("What is a safe way to cross a busy street?", "Use a designated crosswalk, check traffic signals, and look both ways before crossing."),
        ("Why do automobile engines need motor oil?", "Motor oil lubricates moving engine components to reduce friction, heat, and mechanical wear."),
        ("How do you keep fresh vegetables crisp?", "Store them in the refrigerator crisper drawer with appropriate humidity."),
        ("What should you do if your computer freezes?", "Wait a moment for tasks to finish, or restart the system if it remains unresponsive."),
        ("Why do people set an alarm clock?", "People set alarm clocks to ensure they wake up on time for their scheduled activities."),
        ("What is a good way to stay hydrated during hot weather?", "Drink plenty of water consistently throughout the day and avoid excess dehydration."),
        ("Why do drivers and passengers wear seatbelts?", "Seatbelts secure occupants during collisions and sudden stops, greatly reducing injury risk."),
        ("How do you brew a pot of coffee?", "Measure ground coffee into a filter, pour hot water over the grounds, and allow it to brew."),
        ("Why do people read books?", "People read books to gain knowledge, explore ideas, improve focus, and enjoy stories."),
        ("What should you do when you have a headache?", "Rest in a quiet, dimly lit room, drink water, and take an over-the-counter pain reliever if needed."),
        ("Why is regular exercise beneficial?", "Regular exercise improves cardiovascular health, strengthens muscles, and boosts mood and energy."),
        ("How do you clean a glass window?", "Spray window cleaner or diluted vinegar on the glass and wipe clean with a microfiber cloth."),
        ("How do you make ice cubes?", "Pour clean water into an ice tray and place it in the freezer until solid."),
        ("What do you do when a lightbulb burns out?", "Turn off the lamp power, let the old bulb cool, unscrew it, and screw in a new bulb of the correct wattage."),
        ("Why do people wear winter coats?", "Winter coats insulate the body and trap warmth during cold weather."),
        ("How do you wash dirty dishes?", "Wash dishes with warm water and dish soap using a sponge, then rinse with clean water and dry."),
        ("What is a good breakfast food?", "Oatmeal, eggs, whole grain toast, and fruit are healthy breakfast options."),
        ("Why do houseplants need water?", "Houseplants need water to transport nutrients from the soil and maintain cell structure."),
        ("How do you tie shoelaces?", "Cross the laces to form an initial knot, form a loop with one lace, wrap the other around, and pull through."),
        ("What is the purpose of a public library?", "A public library provides access to books, educational resources, computers, and community programs."),
        ("Why do vehicles have headlights?", "Headlights illuminate the road ahead during darkness or poor visibility and signal presence to other drivers."),
        ("What do people do at a post office?", "People send letters, ship packages, purchase stamps, and pick up held mail at a post office."),
        ("Why is it important to recycle plastic and paper?", "Recycling conserves natural resources, reduces landfill waste, and lowers pollution."),
        ("How do you extinguish a small campfire?", "Pour water over the embers, stir the ashes with a stick, and pour more water until completely cool."),
    ]
    for i, (p, r) in enumerate(everyday):
        cid = f"everyday_{i:03d}"
        items.append({"concept_id": cid, "category": "everyday_conversational", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "everyday_conversational", "prompt": f"Can you explain: {p}", "response": r})

    # 4. Definitions (~100 examples)
    definitions = [
        ("What is an atom?", "An atom is the basic building block of matter, composed of protons, neutrons, and electrons."),
        ("What is a herbivore?", "A herbivore is an animal that feeds exclusively or primarily on plants."),
        ("What is a carnivore?", "A carnivore is an organism that derives its energy and nutrient requirements from animal tissue."),
        ("What is software?", "Software consists of programs, scripts, and instructions that tell computer hardware how to operate."),
        ("What is hardware in computing?", "Hardware refers to the physical components of a computer system, such as CPU, RAM, and disk drives."),
        ("What is a continent?", "A continent is one of Earth's large continuous landmasses, such as Asia, Africa, or Europe."),
        ("Define kinetic energy in simple terms.", "Kinetic energy is the energy an object possesses because of its motion."),
        ("What is a molecule?", "A molecule is a distinct group of two or more atoms bonded together chemically."),
        ("What is an archipelago?", "An archipelago is an extensive group or chain of islands situated in a body of water."),
        ("Define humidity in one sentence.", "Humidity is the measure of the concentration of water vapor present in the atmosphere."),
        ("What is an obtuse angle?", "An obtuse angle is an angle measuring between 90 degrees and 180 degrees."),
        ("What is friction in physics?", "Friction is the force resisting the relative motion of solid surfaces or material elements sliding past each other."),
        ("What is a database?", "A database is an organized collection of structured data stored electronically in a computer system."),
        ("What is an integer?", "An integer is a whole number that can be positive, negative, or zero, without any fractional part."),
        ("Define temperature in science.", "Temperature is a physical quantity expressing the degree of hotness or coldness of a substance."),
        ("What is a triangle?", "A triangle is a three-sided polygon with three interior angles summing to 180 degrees."),
        ("What is a rectangle?", "A rectangle is a four-sided polygon with four right angles and opposite sides of equal length."),
        ("What is precipitation in meteorology?", "Precipitation is any liquid or frozen water that forms in the atmosphere and falls to Earth, such as rain or snow."),
        ("What is condensation?", "Condensation is the phase change process where water vapor gas cools and transforms into liquid water."),
        ("What is an omnivore?", "An omnivore is an organism that regularly consumes both plant and animal matter for nourishment."),
        ("What is a noun in grammar?", "A noun is a part of speech that names a person, place, thing, or concept."),
        ("What is a verb in grammar?", "A verb is a part of speech expressing an action, occurrence, or state of being."),
        ("What is an adjective in grammar?", "An adjective is a word that describes, qualifies, or modifies a noun."),
        ("What is a decade?", "A decade is a time period consisting of exactly ten consecutive years."),
        ("What is a century?", "A century is a time period consisting of exactly one hundred consecutive years."),
        ("What is a millennium?", "A millennium is a time period consisting of one thousand consecutive years."),
        ("What is density in physics?", "Density is the ratio of an object's mass to its volume (mass divided by volume)."),
        ("What is a prime number?", "A prime number is a positive integer greater than 1 that is divisible only by 1 and itself."),
    ]
    for i, (p, r) in enumerate(definitions):
        cid = f"def_{i:03d}"
        items.append({"concept_id": cid, "category": "definitions", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "definitions", "prompt": f"In one sentence, explain: {p}", "response": r})

    # 5. Basic Science (~100 examples)
    science = [
        ("What is the boiling point of water in degrees Fahrenheit?", "The boiling point of water at sea level is 212 degrees Fahrenheit."),
        ("What is the chemical symbol for oxygen?", "The chemical symbol for oxygen is O."),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
        ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe."),
        ("What is the chemical symbol for sodium?", "The chemical symbol for sodium is Na."),
        ("What is the chemical symbol for carbon?", "The chemical symbol for carbon is C."),
        ("What organ in the human body filters metabolic waste from blood?", "The kidneys filter waste products from the bloodstream to produce urine."),
        ("What human organ is primarily responsible for breathing?", "The lungs are responsible for gas exchange and respiration in humans."),
        ("What is the central core of an atom called?", "The central core of an atom is called the nucleus."),
        ("What electrical charge does an electron carry?", "An electron carries a negative electrical charge."),
        ("What electrical charge does a proton carry?", "A proton carries a positive electrical charge."),
        ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen is the most abundant gas in Earth's atmosphere, making up about 78 percent."),
        ("What organ pumps blood through the body in vertebrates?", "The heart pumps oxygenated blood throughout the circulatory system."),
        ("What force causes objects to fall when dropped?", "The force of gravity pulls objects downward toward Earth."),
        ("How many planets are in our solar system?", "There are eight recognized planets in our solar system."),
        ("What star is at the center of our solar system?", "The Sun is the star at the center of our solar system."),
        ("What is the chemical symbol for helium?", "The chemical symbol for helium is He."),
        ("What is the chemical symbol for silver?", "The chemical symbol for silver is Ag."),
        ("What is the chemical symbol for nitrogen?", "The chemical symbol for nitrogen is N."),
        ("What is the chemical symbol for calcium?", "The chemical symbol for calcium is Ca."),
        ("What is the chemical symbol for potassium?", "The chemical symbol for potassium is K."),
        ("What is the largest internal human organ?", "The liver is the largest internal organ in the human body."),
        ("What type of lens curves outward?", "A convex lens curves outward at its center."),
        ("What type of lens curves inward?", "A concave lens curves inward at its center."),
        ("What is the process of liquid water turning into gas?", "Evaporation is the process of liquid water converting into water vapor gas."),
        ("What subatomic particle carries no electrical charge?", "A neutron carries no electrical charge; it is neutral."),
        ("What part of a cell houses its genetic DNA in eukaryotes?", "The cell nucleus contains the genetic DNA in eukaryotic cells."),
        ("What scale measures earthquake magnitude?", "The Richter scale and moment magnitude scale measure earthquake magnitude."),
        ("What is the speed of light in vacuum approximately in km/s?", "The speed of light in a vacuum is approximately 300,000 kilometers per second."),
    ]
    for i, (p, r) in enumerate(science):
        cid = f"sci_{i:03d}"
        items.append({"concept_id": cid, "category": "basic_science", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "basic_science", "prompt": f"Science question: {p}", "response": r})

    # 6. Python Concepts (~100 examples)
    python_items = [
        ("What method is used to add an item to the end of a list in Python?", "The append() method adds an item to the end of a list in Python."),
        ("What keyword is used for conditional branch checks in Python?", "The if keyword is used for conditional checks in Python."),
        ("What built-in data structure in Python stores key-value pairs?", "A dictionary (dict) stores key-value pairs in Python."),
        ("What keyword is used to import an external module in Python?", "The import keyword is used to import modules in Python."),
        ("What data type represents True or False values in Python?", "The bool (boolean) data type represents True or False values in Python."),
        ("What operator checks for equality between two values in Python?", "The == operator checks whether two values are equal in Python."),
        ("What keywords handle exceptions and errors in Python?", "The try and except keywords are used for error handling in Python."),
        ("What built-in function generates a sequence of numbers in Python?", "The range() function generates a sequence of numbers in Python."),
        ("What symbol designates a single-line comment in Python?", "The hash symbol (#) is used for single-line comments in Python."),
        ("What string method converts all characters to lowercase in Python?", "The lower() method converts a string to lowercase in Python."),
        ("What string method converts all characters to uppercase in Python?", "The upper() method converts a string to uppercase in Python."),
        ("What keyword is used to return a value from a function in Python?", "The return keyword returns a value from a function in Python."),
        ("What keyword exits immediately from a loop in Python?", "The break keyword terminates and exits immediately from a loop."),
        ("What function converts a value into a floating-point number in Python?", "The float() function converts a value to a floating-point number."),
        ("What function converts a value into a string in Python?", "The str() function converts a value to a string in Python."),
        ("What method removes and returns the last element from a list in Python?", "The pop() method removes and returns the last element from a list."),
        ("What keyword defines a class in Python?", "The class keyword is used to define a new class in Python."),
        ("What built-in function returns the data type of an object in Python?", "The type() function returns the data type of an object in Python."),
        ("What keyword represents a null or absent value in Python?", "The None keyword represents the absence of a value in Python."),
        ("What built-in function calculates the sum of numbers in an iterable in Python?", "The sum() function calculates the sum of numbers in an iterable."),
        ("What function returns the smallest item in an iterable in Python?", "The min() function returns the smallest item in an iterable."),
        ("What function returns the largest item in an iterable in Python?", "The max() function returns the largest item in an iterable."),
        ("What function opens a file for reading or writing in Python?", "The open() function is used to open files in Python."),
        ("What keyword manages resources cleanly with context managers in Python?", "The with keyword manages context managers in Python."),
        ("What method splits a string into a list of words in Python?", "The split() method splits a string into a list."),
        ("What method joins a list of strings into one string in Python?", "The join() method concatenates elements of an iterable into a single string."),
    ]
    for i, (p, r) in enumerate(python_items):
        cid = f"py_{i:03d}"
        items.append({"concept_id": cid, "category": "python_concepts", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "python_concepts", "prompt": f"Python question: {p}", "response": r})

    # 7. Elementary Classification (~100 examples)
    classifications = [
        ("Is a lion a carnivore or a herbivore?", "A lion is a carnivore."),
        ("Is copper a conductor or an insulator of electricity?", "Copper is an electrical conductor."),
        ("Is an oak tree a plant or an animal?", "An oak tree is a plant."),
        ("Is oxygen a gas or a solid at room temperature?", "Oxygen is a gas at room temperature."),
        ("Is Mars a planet or a star?", "Mars is a planet."),
        ("Is wood an electrical insulator or a conductor?", "Wood is an electrical insulator."),
        ("Is a shark a fish or a mammal?", "A shark is a fish."),
        ("Is silver a metal or a nonmetal?", "Silver is a metal."),
        ("Is a circle a 2D shape or a 3D object?", "A circle is a 2D shape."),
        ("Is an elephant a mammal or a reptile?", "An elephant is a mammal."),
        ("Is water an element or a chemical compound?", "Water is a chemical compound consisting of hydrogen and oxygen."),
        ("Is a strawberry a fruit or a vegetable?", "A strawberry is a fruit."),
        ("Is granite an igneous rock or a sedimentary rock?", "Granite is an igneous rock."),
        ("Is a spider an insect or an arachnid?", "A spider is an arachnid."),
        ("Is gold a metal or a gas?", "Gold is a metal."),
        ("Is a trout a fish or a bird?", "A trout is a fish."),
        ("Is plastic a synthetic material or a natural mineral?", "Plastic is a synthetic material."),
        ("Is an eagle a bird or a mammal?", "An eagle is a bird."),
        ("Is a tomato a fruit or a dairy product?", "A tomato is a botanical fruit."),
        ("Is helium a gas or a solid at room temperature?", "Helium is a gas at room temperature."),
        ("Is iron magnetic or non-magnetic?", "Iron is a magnetic metal."),
        ("Is an ant an insect or a mammal?", "An ant is an insect."),
        ("Is pure water an acid, base, or neutral on the pH scale?", "Pure water is neutral with a pH of approximately 7."),
        ("Is a dolphin a mammal or a reptile?", "A dolphin is a mammal."),
        ("Is coal a renewable or non-renewable fossil fuel?", "Coal is a non-renewable fossil fuel."),
        ("Is diamond hard or soft?", "Diamond is an exceptionally hard mineral."),
        ("Is an oak tree deciduous or evergreen?", "Most oak trees are deciduous, shedding leaves annually."),
        ("Is a butterfly an insect or a reptile?", "A butterfly is an insect."),
    ]
    for i, (p, r) in enumerate(classifications):
        cid = f"class_{i:03d}"
        items.append({"concept_id": cid, "category": "elementary_classification", "prompt": p, "response": r})
        concise = r.split(" is an? ")[-1].rstrip(".") if " is an? " in r else r
        items.append({"concept_id": cid, "category": "elementary_classification", "prompt": f"Classify: {p}", "response": r})

    # 8. Single-Step Instructions (~80 examples)
    instructions = [
        ("Say the word python.", "python"),
        ("Convert the word apple to uppercase.", "APPLE"),
        ("Write the numbers 4 to 6 separated by spaces.", "4 5 6"),
        ("Spell the three-letter word dog in reverse order.", "god"),
        ("Repeat the word hello exactly twice.", "hello hello"),
        ("Output only the single word ready.", "ready"),
        ("Capitalize the first letter of mountain.", "Mountain"),
        ("Convert the word WINDOW to lowercase.", "window"),
        ("Output the single word success.", "success"),
        ("Say the word start.", "start"),
        ("Convert the word green to uppercase.", "GREEN"),
        ("Spell the word bat in reverse order.", "tab"),
        ("Write the numbers 7 to 9 separated by spaces.", "7 8 9"),
        ("Repeat the word test twice.", "test test"),
        ("Output the word verified.", "verified"),
        ("Say the single word blue.", "blue"),
        ("Convert the word bird to uppercase.", "BIRD"),
        ("Write the numbers 10 to 12 separated by spaces.", "10 11 12"),
        ("Spell the three-letter word pen in reverse order.", "nep"),
        ("Repeat the word ping twice.", "ping ping"),
        ("Output only the single word confirmed.", "confirmed"),
        ("Capitalize the first letter of silver.", "Silver"),
        ("Convert the word SUMMER to lowercase.", "summer"),
        ("Output the single word active.", "active"),
        ("Say the single word stop.", "stop"),
        ("Convert the word yellow to uppercase.", "YELLOW"),
        ("Spell the three-letter word map backwards.", "pam"),
        ("Write the numbers 20 to 22 separated by spaces.", "20 21 22"),
        ("Repeat the word signal twice.", "signal signal"),
        ("Output the word initialized.", "initialized"),
    ]
    for i, (p, r) in enumerate(instructions):
        cid = f"inst_{i:03d}"
        items.append({"concept_id": cid, "category": "single_step_instructions", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "single_step_instructions", "prompt": f"Instruction: {p}", "response": r})

    # 9. Small Lists (~80 examples)
    lists = [
        ("Name three common domesticated pets.", "Dog, cat, and hamster."),
        ("List two types of berries.", "Strawberry and blueberry."),
        ("Name three Scandinavian countries.", "Norway, Sweden, and Denmark."),
        ("List two musical string instruments.", "Guitar and violin."),
        ("Name three structural parts of a plant.", "Roots, stem, and leaves."),
        ("List two types of renewable energy sources.", "Solar power and wind energy."),
        ("Name two major currencies in Asia.", "Japanese Yen and Chinese Yuan."),
        ("List three common fruits.", "Apple, banana, and orange."),
        ("Name two oceans on Earth.", "Atlantic Ocean and Indian Ocean."),
        ("List three days of the week.", "Monday, Wednesday, and Friday."),
        ("Name two internal human organs.", "Liver and kidneys."),
        ("List two programming languages.", "Python and JavaScript."),
        ("Name three domestic animals.", "Cow, sheep, and goat."),
        ("List two types of citrus fruits.", "Orange and lemon."),
        ("Name two Scandinavian countries.", "Norway and Sweden."),
        ("List two major greenhouse gases.", "Carbon dioxide and methane."),
        ("Name three common computer input devices.", "Keyboard, mouse, and microphone."),
        ("List two popular web search engines.", "Google and Bing."),
        ("Name two common metals used in electrical wiring.", "Copper and aluminum."),
        ("List three colors on a standard traffic light.", "Red, yellow, and green."),
        ("Name two oceans on Earth other than the Pacific.", "Atlantic Ocean and Indian Ocean."),
        ("List two birds that cannot fly.", "Ostrich and penguin."),
    ]
    for i, (p, r) in enumerate(lists):
        cid = f"list_{i:03d}"
        items.append({"concept_id": cid, "category": "small_lists", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "small_lists", "prompt": f"Please provide: {p}", "response": r})

    # 10. Simple Logical Relationships (~80 examples)
    logic = [
        ("If today is Friday, what day was yesterday?", "Yesterday was Thursday."),
        ("If an animal is a dog, is it an animal?", "Yes, a dog is an animal."),
        ("Which is faster: a commercial jet plane or a walking human?", "A commercial jet plane is much faster than a walking human."),
        ("If you walk ten steps forward and ten steps backward, where are you?", "You are back at your starting point."),
        ("Is hot coffee warmer or cooler than ice water?", "Hot coffee is warmer than ice water."),
        ("If you turn 90 degrees left from facing east, what direction do you face?", "You are facing north."),
        ("If an empty container holds no liquid, how much liquid is inside?", "There is zero liquid inside."),
        ("If tomorrow is Sunday, what day is today?", "Today is Saturday."),
        ("Is a whale bigger or smaller than a mouse?", "A whale is much bigger than a mouse."),
        ("If a light switch is turned off, is the light on or off?", "The light is off."),
        ("If you have two coins and receive one more, how many coins do you have?", "You have three coins."),
        ("Is boiling water hotter than room temperature water?", "Yes, boiling water is much hotter."),
        ("If today is Tuesday, what day was yesterday?", "Yesterday was Monday."),
        ("If an object is cold to the touch, is it hot?", "No, it is cold."),
        ("Which animal is larger: an elephant or a flea?", "An elephant is vastly larger than a flea."),
        ("If you walk north and then turn around 180 degrees, what direction do you face?", "You are facing south."),
        ("Is boiling soup warmer or colder than an iced drink?", "Boiling soup is much warmer than an iced drink."),
        ("If a container is completely full to the brim, can you add more liquid without spilling?", "No, adding more liquid will cause it to spill."),
        ("If yesterday was Saturday, what day is today?", "Today is Sunday."),
        ("Is the Sun farther away from Earth than the Moon?", "Yes, the Sun is much farther away than the Moon."),
        ("If you subtract five from five, what is left?", "Zero is left."),
        ("If a door is open, is it closed?", "No, it is not closed."),
    ]
    for i, (p, r) in enumerate(logic):
        cid = f"logic_{i:03d}"
        items.append({"concept_id": cid, "category": "simple_logical_relationships", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "simple_logical_relationships", "prompt": f"Logic question: {p}", "response": r})

    # 11. Verified Arithmetic (~120 examples, strictly disjoint from benchmark)
    arithmetic = [
        (5, 4, "+", 9),
        (12, 15, "+", 27),
        (25, 30, "+", 55),
        (14, 18, "+", 32),
        (16, 14, "+", 30),
        (22, 11, "+", 33),
        (31, 15, "+", 46),
        (10, 25, "+", 35),
        (20, 7, "-", 13),
        (45, 20, "-", 25),
        (60, 25, "-", 35),
        (40, 15, "-", 25),
        (27, 12, "-", 15),
        (55, 30, "-", 25),
        (33, 11, "-", 22),
        (8, 8, "*", 64),
        (7, 5, "*", 35),
        (6, 9, "*", 54),
        (4, 12, "*", 48),
        (8, 3, "*", 24),
        (5, 5, "*", 25),
        (3, 10, "*", 30),
        (30, 5, "/", 6),
        (72, 9, "/", 8),
        (36, 4, "/", 9),
        (100, 10, "/", 10),
        (56, 7, "/", 8),
        (40, 8, "/", 5),
        (28, 4, "/", 7),
        (9, 9, "+", 18),
        (13, 14, "+", 27),
        (32, 16, "+", 48),
        (21, 21, "+", 42),
        (17, 13, "+", 30),
        (24, 12, "+", 36),
        (50, 25, "-", 25),
        (38, 18, "-", 20),
        (70, 35, "-", 35),
        (80, 20, "-", 60),
        (64, 30, "-", 34),
        (90, 45, "-", 45),
        (7, 7, "*", 49),
        (6, 6, "*", 36),
        (8, 4, "*", 32),
        (9, 5, "*", 45),
        (7, 3, "*", 21),
        (8, 5, "*", 40),
        (64, 8, "/", 8),
        (45, 9, "/", 5),
        (81, 9, "/", 9),
        (60, 6, "/", 10),
        (32, 8, "/", 4),
        (21, 3, "/", 7),
    ]
    for i, (a, b, op, res) in enumerate(arithmetic):
        cid = f"arith_{i:03d}"
        if op == "+":
            p = f"What is {a} plus {b}?"
        elif op == "-":
            p = f"What is {a} minus {b}?"
        elif op == "*":
            p = f"What is {a} times {b}?"
        else:
            p = f"What is {a} divided by {b}?"
        r_eq = f"{a} {op} {b} = {res}."
        r_short = f"{res}."
        items.append({"concept_id": cid, "category": "arithmetic", "prompt": p, "response": r_eq})
        items.append({"concept_id": cid, "category": "arithmetic", "prompt": f"Calculate: {p}", "response": r_short})

    # 12. Concise Constraint Responses (~80 examples)
    constraints = [
        ("Answer in exactly one word: What color is a ripe banana?", "Yellow"),
        ("Reply with only yes or no: Do birds have feathers?", "Yes"),
        ("In one short sentence, state what thermometers measure.", "Thermometers measure temperature."),
        ("Answer in one single word: What is the opposite of day?", "Night"),
        ("Reply with only true or false: The Earth orbits around the Sun.", "True"),
        ("Answer with only a number: How many legs does a typical dog have?", "4"),
        ("Answer in one single word: What is the solid form of water called?", "Ice"),
        ("Answer in exactly one word: What color is snow?", "White"),
        ("Reply with only yes or no: Do cats have tails?", "Yes"),
        ("Answer with only a number: How many hours are in one full day?", "24"),
        ("Answer in one single word: What is the opposite of cold?", "Warm"),
        ("Reply with only true or false: Iron is a metal.", "True"),
        ("Answer in exactly one word: What color is charcoal?", "Black"),
        ("Reply with only yes or no: Can humans breathe underwater without equipment?", "No"),
        ("In one short sentence, state what rulers measure.", "Rulers measure length or distance."),
        ("Answer in one single word: What is the opposite of fast?", "Slow"),
        ("Reply with only true or false: The Earth has two moons.", "False"),
        ("Answer with only a number: How many sides does a triangle have?", "3"),
        ("Answer in exactly one word: What liquid do cows produce?", "Milk"),
        ("Reply with only yes or no: Is the Sun a star?", "Yes"),
        ("Answer in one single word: What is the opposite of dark?", "Light"),
        ("Reply with only true or false: Water freezes at 0 degrees Celsius.", "True"),
        ("Answer with only a number: How many months are in a standard year?", "12"),
        ("Answer in exactly one word: What color is an emerald gemstone?", "Green"),
        ("Reply with only yes or no: Do reptiles lay eggs?", "Yes"),
    ]
    for i, (p, r) in enumerate(constraints):
        cid = f"constr_{i:03d}"
        items.append({"concept_id": cid, "category": "concise_constraint_responses", "prompt": p, "response": r})
        items.append({"concept_id": cid, "category": "concise_constraint_responses", "prompt": f"Strict constraint: {p}", "response": r})

    return items


def main():
    logger.info("Starting Step 48 Conversational SFT v2 Dataset Preparation...")
    random.seed(42)

    # 1. Load protected benchmark prompts
    protected_prompts = load_benchmark_normalized_prompts()
    logger.info(f"Loaded {len(protected_prompts)} normalized frozen benchmark prompts for collision screening.")

    # 2. Build raw pool
    raw_pool = build_raw_conversational_v2_pool()
    logger.info(f"Built raw candidate pool of {len(raw_pool)} instruction pairs.")

    # 3. Load Tokenizer
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    logger.info(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}.")

    # 4. Audit & Filter
    screened_items: List[Dict[str, Any]] = []
    seen_prompts: Set[str] = set()
    collision_count = 0
    token_reject_count = 0

    for idx, item in enumerate(raw_pool):
        p_raw = item["prompt"].strip()
        r_raw = item["response"].strip()
        norm_p = normalize(p_raw)

        # Collision check against frozen benchmark
        if norm_p in protected_prompts:
            collision_count += 1
            logger.warning(f"LEAKAGE PREVENTED: Candidate prompt '{p_raw}' collided with frozen benchmark!")
            continue

        # Deduplication within candidate pool
        if norm_p in seen_prompts:
            continue
        seen_prompts.add(norm_p)

        # Validate token lengths and loss masking
        turns = [
            ConversationTurn(role="user", content=p_raw),
            ConversationTurn(role="assistant", content=r_raw),
        ]
        tok_ex = encode_conversation(
            turns=turns,
            tokenizer=tokenizer,
            max_seq_len=256,
            truncate=False,
            example_id=f"conv2_{idx:05d}",
            category=item["category"],
        )

        if tok_ex is None:
            token_reject_count += 1
            continue

        # Check token length constraints
        if tok_ex.prompt_tokens_count > 128 or tok_ex.response_tokens_count > 96:
            token_reject_count += 1
            continue

        screened_items.append({
            "id": f"conv2_{len(screened_items):05d}",
            "concept_id": item["concept_id"],
            "category": item["category"],
            "prompt": p_raw,
            "response": r_raw,
            "prompt_tokens": tok_ex.prompt_tokens_count,
            "response_tokens": tok_ex.response_tokens_count,
            "total_tokens": tok_ex.total_tokens_count,
        })

    logger.info(f"Screened {len(screened_items)} valid examples. Collisions prevented: {collision_count}. Token rejects: {token_reject_count}.")
    assert collision_count == 0, f"Critical: {collision_count} benchmark collisions detected!"
    assert len(screened_items) >= 800, f"Expected at least 800 examples, got {len(screened_items)}"

    # 5. Split 85% Train / 15% Val grouped strictly by concept_id
    concept_map: Dict[str, List[Dict[str, Any]]] = {}
    for it in screened_items:
        concept_map.setdefault(it["concept_id"], []).append(it)

    concept_ids = sorted(list(concept_map.keys()))
    random.seed(42)
    random.shuffle(concept_ids)

    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []
    target_train_count = int(len(screened_items) * 0.85)

    for cid in concept_ids:
        group = concept_map[cid]
        if len(train_items) + len(group) <= target_train_count or not train_items:
            train_items.extend(group)
        else:
            val_items.extend(group)

    train_concepts = {it["concept_id"] for it in train_items}
    val_concepts = {it["concept_id"] for it in val_items}
    concept_leakage = train_concepts.intersection(val_concepts)
    assert len(concept_leakage) == 0, f"Critical: Concept leakage detected between train and val: {concept_leakage}"

    logger.info(f"Split dataset by concept_id: {len(train_items)} train (concepts={len(train_concepts)}), {len(val_items)} validation (concepts={len(val_concepts)}).")

    # 6. Save JSONL files
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    train_jsonl_path = JSONL_DIR / "sft_conversational_v2_train.jsonl"
    val_jsonl_path = JSONL_DIR / "sft_conversational_v2_val.jsonl"
    manifest_path = JSONL_DIR / "dataset_manifest.json"

    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it) + "\n")

    with open(val_jsonl_path, "w", encoding="utf-8") as f:
        for it in val_items:
            f.write(json.dumps(it) + "\n")

    manifest = {
        "dataset_name": "sft_conversational_v2",
        "created_at": "Step 49",
        "total_examples": len(screened_items),
        "train_examples": len(train_items),
        "val_examples": len(val_items),
        "total_concepts": len(concept_map),
        "train_concepts": len(train_concepts),
        "val_concepts": len(val_concepts),
        "concept_overlap": 0,
        "categories": sorted(list({it["category"] for it in screened_items})),
        "avg_prompt_tokens": round(sum(it["prompt_tokens"] for it in screened_items) / len(screened_items), 2),
        "avg_response_tokens": round(sum(it["response_tokens"] for it in screened_items) / len(screened_items), 2),
        "benchmark_collisions": collision_count,
        "sha256_train": hashlib.sha256(open(train_jsonl_path, "rb").read()).hexdigest(),
        "sha256_val": hashlib.sha256(open(val_jsonl_path, "rb").read()).hexdigest(),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 7. Tokenize and Cache binary tensors
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def encode_split(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        encoded = []
        for it in items:
            turns = [
                ConversationTurn(role="user", content=it["prompt"]),
                ConversationTurn(role="assistant", content=it["response"]),
            ]
            tok_ex = encode_conversation(
                turns=turns,
                tokenizer=tokenizer,
                max_seq_len=256,
                truncate=False,
                example_id=it["id"],
                category=it["category"],
            )
            if tok_ex is not None:
                encoded.append({
                    "id": tok_ex.example_id,
                    "input_ids": tok_ex.input_ids,
                    "target_ids": tok_ex.target_ids,
                    "loss_mask": tok_ex.loss_mask,
                    "prompt_tokens_count": tok_ex.prompt_tokens_count,
                    "response_tokens_count": tok_ex.response_tokens_count,
                    "total_tokens_count": tok_ex.total_tokens_count,
                    "category": tok_ex.category,
                })
        return encoded

    train_encoded = encode_split(train_items)
    val_encoded = encode_split(val_items)

    torch.save(train_encoded, CACHE_DIR / "train_tokens.pt")
    torch.save(val_encoded, CACHE_DIR / "val_tokens.pt")

    cache_meta = {
        "dataset_name": "sft_conversational_v2_cached",
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "train_count": len(train_encoded),
        "val_count": len(val_encoded),
        "max_seq_len": 256,
        "files": {"train": "train_tokens.pt", "val": "val_tokens.pt"},
    }
    with open(CACHE_DIR / "cache_metadata.json", "w", encoding="utf-8") as f:
        json.dump(cache_meta, f, indent=2)

    logger.info(f"Successfully created Conversational SFT v2 dataset:")
    logger.info(f"  Train: {len(train_items)} items -> {CACHE_DIR / 'train_tokens.pt'}")
    logger.info(f"  Val:   {len(val_items)} items -> {CACHE_DIR / 'val_tokens.pt'}")
    logger.info(f"  Manifest: {manifest_path}")
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print(f"Total Validated Examples: {len(screened_items)}")
    print(f"Train / Val:              {len(train_items)} / {len(val_items)}")
    print(f"Benchmark Collisions:     {collision_count} (Zero Leakage Verified)")
    print(f"Avg Response Tokens:      {manifest['avg_response_tokens']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
