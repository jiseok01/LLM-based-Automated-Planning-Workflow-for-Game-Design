from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from konlpy.tag import Okt
import torch
import re
from typing import List
from itertools import combinations

### 모델 준비
sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

nli_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
nli_model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-discriminator")
nli_classifier = pipeline("text-classification", model=nli_model, tokenizer=nli_tokenizer)

okt = Okt()
conflict_keywords = ["하지만", "그러나", "위협", "방해", "두려움", "갈등", "적", "문제", "충돌", "장애물"]

### 함수 정의
def get_similarity(text1: str, text2: str) -> float:
    emb1 = sbert_model.encode(text1, convert_to_tensor=True)
    emb2 = sbert_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def check_contradiction(premise: str, hypothesis: str) -> str:
    inputs = nli_tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
    outputs = nli_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    labels = ['entailment', 'neutral', 'contradiction']
    return labels[probs.argmax()]

def extract_entities(text: str) -> List[str]:
    return list(set(okt.nouns(text)))

def compare_entities(ent1: List[str], ent2: List[str]) -> List[str]:
    return list(set(ent1) & set(ent2))

def has_conflict(text: str) -> bool:
    return any(k in text for k in conflict_keywords)

### 챕터 내부 비교
def analyze_chapter_internal(chapter: str) -> None:
    print("[챕터 내부 점검]")
    sentences = [s.strip() for s in re.split(r'[\n.?!]', chapter) if len(s.strip()) > 5]

    contradictions = []
    sim_scores = []

    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        sim = get_similarity(s1, s2)
        contradiction = check_contradiction(s1, s2)
        sim_scores.append(sim)
        if contradiction == "contradiction":
            contradictions.append((s1, s2))

    print(f" - 평균 문장 유사도: {sum(sim_scores)/len(sim_scores):.2f}")
    print(f" - 갈등 포함 여부: {'있음' if has_conflict(chapter) else '없음'}")
    print(f" - 내부 모순 {len(contradictions)}건 발견")
    for s1, s2 in contradictions:
        print(f"   모순 문장: '{s1}' ⟷ '{s2}'")

### 챕터 간 비교
def analyze_chapters(chapters: List[str]) -> None:
    for idx, chapter in enumerate(chapters):
        print(f"\n========== 챕터 {idx+1} ==========")
        analyze_chapter_internal(chapter)

    print("\n📘 [챕터 간 비교 - 전체 조합]")
    for i, j in combinations(range(len(chapters)), 2): 
        print(f"\n [챕터 {i+1} ↔ 챕터 {j+1} 비교]")
        sim = get_similarity(chapters[i], chapters[j])
        contradiction = check_contradiction(chapters[i], chapters[j])
        ent1 = extract_entities(chapters[i])
        ent2 = extract_entities(chapters[j])
        common_ents = compare_entities(ent1, ent2)

        print(f" - 의미 유사도: {sim:.2f}")
        print(f" - 논리 관계 (NLI): {contradiction}")
        print(f" - 공통 엔티티: {common_ents}")

def split_markdown_chapters(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'(^##\s*\d+\..*?)(?=^##\s*\d+\.|\Z)', re.DOTALL | re.MULTILINE)
    matches = pattern.findall(content)

    chapters = [match.strip() for match in matches]
    return chapters

chapters = split_markdown_chapters("") #md파일 경로

analyze_chapters(chapters)