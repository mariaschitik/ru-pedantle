import json
import re
from gensim.models import KeyedVectors
from pymorphy2 import MorphAnalyzer


# Загружаем векторную модель
model = KeyedVectors.load_word2vec_format('model.txt', binary=False)

# Инициализируем морфоанализатор
morph = MorphAnalyzer()

def load_data(filename='corpus_lemmas.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_titles(filename='titles_lemmas.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_links(filename='all_links.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.read().split('\n') if line.strip()]
        return lines[::2], lines[1::2]  # названия, ссылки


def mask_text(words, revealed_indices):
    result = []
    for i, word in enumerate(words):
        if word.strip() == '':
            result.append(word)
        elif re.match(r'^\W+$', word):
            result.append(word)
        elif i in revealed_indices:
            result.append(word)
        else:
            result.append('_' * len(word))
    return ''.join(result)


def get_all_possible_keys(word):
    parsed = morph.parse(word)
    keys = set()
    for p in parsed:
        lemma = p.normal_form
        pos = p.tag.POS  # Например, 'NOUN', 'VERB'
        if lemma and pos:
            keys.add(f"{lemma}_{pos}")
    return keys


def find_nearest_word(user_input, text_lemmas, threshold=0.3):
    possible_keys = get_all_possible_keys(user_input)
    print(f"[DEBUG] Возможные ключи для '{user_input}': {possible_keys}")

    if not possible_keys:
        print(f"[DEBUG] '{user_input}' не имеет ключей.")
        return None

    # подготовим ключи для текста
    text_keys = prepare_text_keys(text_lemmas)

    best_match = None
    best_score = threshold

    for input_key in possible_keys:
        if input_key not in model:
            continue
        for i, (lemma, text_key) in enumerate(text_keys):
            if text_key not in model:
                continue
            sim = model.similarity(input_key, text_key)
            if sim > best_score:
                best_score = sim
                best_match = (i, lemma, sim)

    if best_match:
        print(f"[DEBUG] Нашли похожее: {user_input} → {best_match[1]} (score={best_match[2]:.2f})")
        return best_match
    else:
        print(f"[DEBUG] Нет близких слов для '{user_input}' выше порога {threshold}")
        return None


def prepare_text_keys(text_lemmas):
    keys = []
    for lemma in text_lemmas:
        parsed = morph.parse(lemma)
        if parsed:
            best = parsed[0]
            pos = best.tag.POS
            if pos:
                key = f"{best.normal_form}_{pos}"
                if key in model:
                    keys.append((lemma, key))
    return keys



def play_article(index, data, titles_data, article_links):
    first_article = data[index]
    original_words = first_article['original_words']
    lemmas = first_article['lemmas']

    title_text = titles_data[index]['title']
    title_lemmas = titles_data[index]['lemmas']
    link = article_links[index]

    title_words = re.findall(r'\w+|\W+', title_text)
    title_indices = set(range(len(title_words)))
    revealed_indices = set()

    print("\n" + "="*50)
    print(f"\nИГРА №{index+1}")
    print("\nНазвание статьи:")
    print(mask_text(title_words, revealed_indices))
    print("\nТекст с закрытыми словами:")
    print(mask_text(original_words, revealed_indices))

    while True:
        guess = input("\nВведите лемму или слово для открытия (или 'exit' для выхода): ").strip().lower()
        if guess == 'exit':
            print("Спасибо за игру!")
            return False

                # Ищем совпадения в тексте
        matches = [
            i for i, (lemma, orig) in enumerate(zip(lemmas, original_words))
            if lemma.lower() == guess or orig.lower() == guess
        ]

        # Ищем совпадения в заголовке
        title_matches = [
            i for i in title_indices
            if morph.parse(title_words[i])[0].normal_form.lower() == guess
            or title_words[i].lower() == guess
        ]

        if matches or title_matches:
            revealed_indices.update(matches)
            revealed_indices.update(title_matches)

            print("\nНазвание статьи:")
            print(mask_text(title_words, revealed_indices))

            print("\nТекст с закрытыми словами:")
            print(mask_text(original_words, revealed_indices))

            # Проверяем, раскрыты ли леммы заголовка именно в заголовке
            revealed_title_lemmas = [
                morph.parse(title_words[i])[0].normal_form.lower()
                for i in revealed_indices
                if i in title_indices and title_words[i].strip().isalpha()
            ]

            if all(lemma.lower() in revealed_title_lemmas for lemma in title_lemmas):
                revealed_indices.update(title_indices)

                print("\n🔥 Победа! Вы угадали название:", title_text)
                print("📌 Ссылка:", link)

                print("\nПолный текст статьи:")
                print(''.join(original_words))

                print("\nНазвание статьи:")
                print(title_text)

                choice = input("\nХотите сыграть ещё один раунд? (да/нет): ").strip().lower()
                return choice == 'да'

        else:
            nearest = find_nearest_word(guess, lemmas)
            if nearest:
                i, lemma, score = nearest
                print(f"Слова '{guess}' нет в тексте, но оно близко к слову №{i} ('{original_words[i]}') с похожестью {score:.2f}.")
            else:
                print("Такой леммы или слова в тексте нет. Попробуйте еще.")




def main():
    data = load_data()
    titles_data = load_titles()
    article_titles, article_links = load_links()

    index = 0
    n = len(data)

    while index < n:
        print(f"\nТекущий номер статьи: {index + 1} из {n}")
        print("Вы можете ввести:")
        print("  - 'продолжить' для игры с текущей статьи")
        print("  - номер статьи (например, 5), чтобы перейти к ней")
        print("  - 'exit' для выхода")

        command = input("\nВаш ввод: ").strip().lower()

        if command == 'exit':
            print("Спасибо за игру!")
            break

        elif command == 'продолжить':
            continue_game = play_article(index, data, titles_data, article_links)
            if continue_game:
                index += 1
            else:
                break

        elif command.isdigit():
            new_index = int(command) - 1
            if 0 <= new_index < n:
                index = new_index
            else:
                print(f"Ошибка: введите число от 1 до {n}.")
        else:
            print("Некорректная команда. Попробуйте снова.")

    print("\nИгра завершена. Спасибо за участие!")


if __name__ == '__main__':
    main()
