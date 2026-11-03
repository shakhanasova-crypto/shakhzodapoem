import streamlit as st
import pandas as pd
import re
from collections import Counter
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Doc

VOWELS = "аеёиоуыэюя"
POS_RU = {"NOUN": "существительное", "VERB": "глагол", "ADJ": "прилагательное", "ADV": "наречие",
          "PRON": "местоимение", "ADP": "предлог", "CCONJ": "союз", "SCONJ": "союз",
          "PART": "частица", "DET": "местоимение-прил.", "PROPN": "имя собственное",
          "NUM": "числительное", "AUX": "вспом. глагол", "INTJ": "междометие"}

# ---------- базовый анализ ----------
def load_models():
    emb = NewsEmbedding()
    return dict(seg=Segmenter(), vocab=MorphVocab(), morph=NewsMorphTagger(emb), ner=NewsNERTagger(emb))

def split_poem(text):
    stanzas = [s for s in re.split(r"\n\s*\n", text.strip()) if s.strip()]
    return [[l.strip() for l in s.split("\n") if l.strip()] for s in stanzas]

def basic_stats(stanzas):
    lines = [l for s in stanzas for l in s]
    words = re.findall(r"[а-яёА-ЯЁa-zA-Z-]+", " ".join(lines))
    return {"строф": len(stanzas), "строк": len(lines), "слов": len(words),
            "уникальных слов": len({w.lower() for w in words}),
            "слов в строке (среднее)": round(len(words) / max(len(lines), 1), 1)}

def syllables(line):
    return sum(ch in VOWELS for ch in line.lower())

def morph_analysis(text, m):
    doc = Doc(text)
    doc.segment(m["seg"]); doc.tag_morph(m["morph"]); doc.tag_ner(m["ner"])
    rows = []
    for t in doc.tokens:
        if not re.search(r"[а-яёa-z]", t.text.lower()):
            continue
        t.lemmatize(m["vocab"])
        rows.append({"слово": t.text, "лемма": t.lemma, "часть речи": POS_RU.get(t.pos, t.pos)})
    ents = []
    for s in doc.spans:
        s.normalize(m["vocab"])
        ents.append({"тип": {"PER": "персона", "LOC": "место", "ORG": "организация"}[s.type], "имя": s.normal})
    return rows, ents

# ---------- рифма без ударений ----------
def last_word(line):
    words = re.findall(r"[а-яёa-z]+", line.lower())
    return words[-1].replace("ё", "е") if words else ""

def rhyme_keys(word):
    pos = [i for i, ch in enumerate(word) if ch in VOWELS]
    if not pos:
        return None, word
    strong = None
    if len(pos) >= 2:
        tail = word[pos[-2]:]
        strong = tail[0] + "".join("*" if ch in VOWELS else ch for ch in tail[1:])
    last = pos[-1]
    weak = word[last:]
    if last == len(word) - 1 and last > 0 and word[last - 1] not in VOWELS:
        weak = word[last - 1:]
    return strong, weak

def is_rhyme(w1, w2):
    if not w1 or not w2 or w1 == w2:
        return w1 == w2 and bool(w1)
    s1, k1 = rhyme_keys(w1); s2, k2 = rhyme_keys(w2)
    return (s1 is not None and s1 == s2) or k1 == k2

def rhyme_scheme(stanza):
    letters, ends = [], []
    abc = "АБВГДЕЖЗИКЛМН"
    for line in stanza:
        w = last_word(line)
        for j, prev in enumerate(ends):
            if is_rhyme(w, prev):
                letters.append(letters[j]); break
        else:
            letters.append(abc[len(set(letters))])
        ends.append(w)
    return letters

# ---------- стихотворный размер ----------
METRES = {"хорей": (2, 0), "ямб": (2, 1), "дактиль": (3, 0), "амфибрахий": (3, 1), "анапест": (3, 2)}
WEAK_WORDS = {"передо", "перед", "через", "между", "около", "после", "кроме", "среди", "сквозь",
              "чтобы", "если", "будто", "словно", "только", "даже", "или", "либо", "когда", "пока",
              "над", "надо", "подо", "ото", "изо", "обо", "чего", "чтоб", "ведь", "уже", "еще", "ещё"}
FEET = {2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти", 6: "шести", 7: "семи", 8: "восьми"}

def line_syllables(stressed_line):
    out = []
    for word in re.findall(r"[а-яёА-ЯЁ+]+", stressed_line):
        w = word.lower()
        plus = [i for i, ch in enumerate(w) if ch == "+"]
        clean = w.replace("+", "")
        vow = [i for i, ch in enumerate(clean) if ch in VOWELS]
        if not vow:
            continue
        stress = None
        if plus:
            p = plus[0]
            stress = sum(1 for i in vow if i < p)
        elif "ё" in clean:
            stress = [clean[i] for i in vow].index("ё")
        if len(vow) == 1 or clean in WEAK_WORDS:
            if len(vow) == 1:
                out.append("s" if stress is not None else "m")
            else:
                out.extend(("s" if k == stress else "u") for k in range(len(vow)))
        else:
            out.extend(("S" if k == stress else "u") if stress is not None else "?" for k in range(len(vow)))
    return out

def score_line(sylls, period, offset):
    strong = lambda i: (i - offset) % period == 0
    good = sum(1 for i, s in enumerate(sylls) if s == "S" and strong(i))
    bad = sum(1 for i, s in enumerate(sylls) if s == "S" and not strong(i))
    light = sum(1 for i, s in enumerate(sylls) if s == "s" and not strong(i))
    return good - 2 * bad - 0.25 * light, bad

def feet_count(sylls, period, offset):
    stressed = [i for i, s in enumerate(sylls) if s in "Ss"]
    if not stressed:
        return None
    return sum(1 for i in range(stressed[-1] + 1) if (i - offset) % period == 0)

def clausula(sylls):
    stressed = [i for i, s in enumerate(sylls) if s in "Ss"]
    if not stressed:
        return None
    tail = len(sylls) - 1 - stressed[-1]
    return {0: "мужская", 1: "женская", 2: "дактилическая"}.get(tail, "гипердактилическая")

def detect_metre(stressed_lines):
    lines = [line_syllables(l) for l in stressed_lines if l.strip()]
    lines = [l for l in lines if l]
    results = []
    for name, (p, o) in METRES.items():
        total, clean_lines = 0, 0
        for sy in lines:
            sc, bad = score_line(sy, p, o)
            total += sc
            clean_lines += (bad == 0)
        results.append({"размер": name, "баллы": round(total, 2),
                        "строк без нарушений": f"{clean_lines} из {len(lines)}",
                        "_share": clean_lines / max(len(lines), 1)})
    results.sort(key=lambda r: -r["баллы"])
    best = results[0]
    p, o = METRES[best["размер"]]
    feet = Counter(f for f in (feet_count(sy, p, o) for sy in lines) if f)
    n = feet.most_common(1)[0][0] if feet else None
    classic = best["_share"] >= 0.75
    if classic and n:
        verdict = f"{n}-стопный {best['размер']}" + (f" ({FEET[n]}стопный)" if n in FEET else "")
    elif classic:
        verdict = best["размер"]
    else:
        verdict = (f"классический размер не определён уверенно (ближе всего — {best['размер']}); "
                   "возможно, дольник, тактовик, вольный стих или ошибки в ударениях")
    clauses = Counter(c for c in (clausula(sy) for sy in lines) if c)
    for r in results:
        r.pop("_share")
    return {"вердикт": verdict, "уверенно": classic, "таблица": results,
            "схемы": ["".join("—" if s in "Ss" else "∪" for s in sy) for sy in lines],
            "клаузулы": dict(clauses)}

# ---------- темы ----------
THEMES = {
    "Любовь и чувства": "любовь любить любимый влюблённый влюбленный сердце нежный нежность страсть страстный поцелуй целовать милый ласка ласковый разлука свидание ревность чувство очи взор",
    "Природа": "лес поле небо солнце луна месяц звезда море река ветер буря снег дождь цветок роза сад трава дерево берёза береза весна лето осень зима гора туча облако заря рассвет закат лист листва птица соловей ручей волна мгла вихрь степь природа роща",
    "Время и память": "время год день век миг мгновение мгновенье помнить память вспоминать воспоминание былой прошлое прежний давно вечность вечный юность молодость старость час минута",
    "Смерть и утрата": "смерть умереть умирать могила гроб похороны мёртвый мертвый погибнуть гибель кладбище хоронить утрата тлен прах",
    "Печаль и одиночество": "грусть грустный тоска печаль печальный одиночество одинокий слеза плакать скорбь горе уныние томление томленье безнадёжный безнадежный тревога страдание страдать боль",
    "Радость и надежда": "радость радостный счастье счастливый восторг надежда весёлый веселый смех улыбка упоение упоенье ликовать пробуждение пробужденье воскреснуть",
    "Вера и Бог": "бог господь божество молитва молиться храм церковь церковный купол хор свеча икона ангел небесный святой грех рай вера крест чудо душа",
    "Родина и народ": "родина отчизна отечество россия русь народ родной край страна земля деревня мужик крестьянин",
    "Творчество и искусство": "поэт поэзия стих песня петь муза вдохновение вдохновенье лира гений слово творец творить звук искусство красота",
    "Свобода и борьба": "свобода воля вольный бой битва война враг цепь оковы плен тюрьма бунт мятежный сражение герой слава власть тиран",
    "Дорога и странствие": "дорога путь странник скитаться даль чужбина изгнанник корабль парус берег",
    "Город": "город улица фонарь аптека площадь переулок трамвай столица",
    "Дом и семья": "дом мать мама отец сын дочь дитя ребёнок ребенок семья очаг колыбель",
    "Ночь, сон, мечта": "ночь сон сниться мечта мечтать тьма видение виденье призрак тень",
}
THEMES = {k: set(v.split()) for k, v in THEMES.items()}

def detect_themes(lemmas):
    c = Counter(l.lower() for l in lemmas)
    rows = []
    for theme, words in THEMES.items():
        hits = {w: c[w] for w in words if c[w]}
        n = sum(hits.values())
        if n:
            rows.append({"тема": theme, "совпадений": n,
                         "слова": ", ".join(f"{w} ({k})" if k > 1 else w for w, k in sorted(hits.items(), key=lambda x: -x[1]))})
    rows.sort(key=lambda r: -r["совпадений"])
    return rows

# ---------- ударения и рифма по ударениям ----------
def show_stress(line):
    return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "\\1\u0301", line)

def stressed_ending(stressed_word):
    w = stressed_word.lower()
    if "+" not in w:
        return None
    tail = w[w.index("+") + 1:].replace("+", "").replace("ё", "е")
    return tail[0] + "".join("*" if ch in VOWELS else ch for ch in tail[1:])

def last_stressed_word(stressed_line):
    words = re.findall(r"[а-яёА-ЯЁ+]+", stressed_line)
    return words[-1] if words else ""

def rhyme_scheme_stressed(stanza, stressed_stanza=None):
    if not stressed_stanza:
        return rhyme_scheme(stanza)
    letters, keys = [], []
    abc = "АБВГДЕЖЗИКЛМН"
    for line, sline in zip(stanza, stressed_stanza):
        k = stressed_ending(last_stressed_word(sline))
        w = last_word(line)
        for j, (pk, pw) in enumerate(keys):
            if (k and pk and k == pk) or ((not k or not pk) and is_rhyme(w, pw)):
                letters.append(letters[j]); break
        else:
            letters.append(abc[len(set(letters))])
        keys.append((k, w))
    return letters

# ================== ИНТЕРФЕЙС САЙТА ==================
st.set_page_config(page_title="Анализ стихотворения", page_icon="📜", layout="wide")
st.title("Анализ стихотворения")
st.caption("Вставьте текст стихотворения. Строфы разделяйте пустой строкой. "
           "Ударения можно поставить самостоятельно знаком + перед ударной гласной (мгнов+енье) — "
           "тогда программа не будет расставлять их сама.")

@st.cache_resource(show_spinner="Загружаю модели Natasha…")
def get_models():
    return load_models()

@st.cache_resource(show_spinner="Загружаю модель ударений (в первый раз 1–3 минуты)…")
def get_accentizer():
    from ruaccent import RUAccent
    acc = RUAccent()
    acc.load(omograph_model_size="turbo3.1", use_dictionary=False)
    return acc

EXAMPLE = """Я помню чудное мгновенье:
Передо мной явилась ты,
Как мимолетное виденье,
Как гений чистой красоты.

В томленьях грусти безнадежной,
В тревогах шумной суеты,
Звучал мне долго голос нежный
И снились милые черты."""

text = st.text_area("Текст стихотворения", EXAMPLE, height=300)

if st.button("Анализировать", type="primary") and text.strip():
    clean_text = text.replace("+", "")
    stanzas = split_poem(clean_text)
    raw_stanzas = split_poem(text)
    models = get_models()

    stressed = None
    if "+" in text:
        stressed = raw_stanzas
        stress_note = "Использованы ударения, расставленные вручную."
    else:
        try:
            acc = get_accentizer()
            stressed = [[acc.process_all(line) for line in stanza] for stanza in stanzas]
            stress_note = "Ударения расставлены автоматически (библиотека ruaccent) и могут содержать ошибки."
        except Exception as e:
            stress_note = f"Не удалось загрузить модель ударений ({type(e).__name__}). Поставьте ударения вручную знаком +."

    tabs = st.tabs(["Общее", "Размер", "Рифма", "Темы", "Лексика", "Имена и места"])

    with tabs[0]:
        stats = basic_stats(stanzas)
        cols = st.columns(len(stats))
        for col, (name, value) in zip(cols, stats.items()):
            col.metric(name, value)

    with tabs[1]:
        st.subheader("Стихотворный размер")
        if stressed:
            all_lines = [l for s in stressed for l in s]
            res = detect_metre(all_lines)
            if res["уверенно"]:
                st.success(f"**{res['вердикт'].capitalize()}**")
            else:
                st.warning(res["вердикт"].capitalize())
            if res["клаузулы"]:
                st.write("Клаузулы (окончания строк):",
                         ", ".join(f"{k} — {v}" for k, v in res["клаузулы"].items()))
            lines_df = pd.DataFrame({"строка с ударениями": [show_stress(l) for l in all_lines if l.strip()][:len(res["схемы"])],
                                     "схема (∪ безударный, — ударный)": res["схемы"]})
            st.dataframe(lines_df, width="stretch", hide_index=True)
            with st.expander("Как программа выбирала размер"):
                st.write("Для каждого размера считается, сколько ударений попало на сильные места (плюс) "
                         "и сколько на слабые (минус). Побеждает размер с наибольшим числом баллов. "
                         "Если меньше 75% строк укладываются в размер без нарушений, программа не делает вывод.")
                st.dataframe(pd.DataFrame(res["таблица"]), width="stretch", hide_index=True)
        st.caption(stress_note)

    with tabs[2]:
        st.subheader("Строки, слоги и рифма")
        rows, schemes = [], []
        for i, stanza in enumerate(stanzas):
            sst = stressed[i] if stressed and i < len(stressed) else None
            scheme = rhyme_scheme_stressed(stanza, sst)
            schemes.append("".join(scheme))
            for line, letter in zip(stanza, scheme):
                rows.append({"строфа": i + 1, "строка": line, "слогов": syllables(line),
                             "последнее слово": last_word(line), "рифма": letter})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.write("Схемы рифмовки по строфам:", " · ".join(schemes))
        st.caption("Рифма определяется по ударному слогу, а если ударений нет — по окончаниям слов.")

    words, ents = morph_analysis(clean_text, models)
    wdf = pd.DataFrame(words)

    with tabs[3]:
        st.subheader("Основные темы")
        th = detect_themes(wdf["лемма"].tolist()) if len(wdf) else []
        main = [t for t in th if t["совпадений"] >= 2]
        if main:
            st.write("**Ведущие темы:** " + ", ".join(t["тема"] for t in main[:3]))
            st.dataframe(pd.DataFrame(th), width="stretch", hide_index=True,
                         column_config={"совпадений": st.column_config.ProgressColumn(
                             "совпадений", format="%d", min_value=0, max_value=max(t["совпадений"] for t in th))})
        else:
            st.write("Явных тем по словарям не найдено.")
        st.caption("Темы определяются по тематическим словарям: программа ищет в стихотворении слова, "
                   "связанные с каждой темой. Это подсказка для анализа, а не окончательный вывод — "
                   "метафоры и подтекст словари не видят.")

    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Части речи")
            st.bar_chart(wdf["часть речи"].value_counts())
        with c2:
            st.subheader("Самые частые значимые слова (леммы)")
            content = wdf[wdf["часть речи"].isin(["существительное", "глагол", "прилагательное", "наречие"])]
            top = content["лемма"].value_counts().head(15).rename_axis("лемма").reset_index(name="раз")
            st.dataframe(top, width="stretch", hide_index=True)

    with tabs[5]:
        if ents:
            st.dataframe(pd.DataFrame(ents), width="stretch", hide_index=True)
            st.caption("В стихах каждая строка начинается с заглавной буквы, поэтому NER иногда "
                       "принимает обычные слова за имена.")
        else:
            st.write("Имён и мест не найдено.")
