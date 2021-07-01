import itertools

from src.data import sub_chapters_toc
from src.data import sub_chapters


def concept_concept_similarity(concept_1, concept_2):
    if concept_1 in concept_2:
        return 1.0
    else:
        return 0.0


def concept_sub_chapter_similarity(concept, sub_chapter):

    if concept in sub_chapter:
        return 1.0
    else:
        return 0.0


def calculate_complexity_level(concept_1, concept_2, concept_sub_chapter_map):

    concept_1_sub_chapter = concept_sub_chapter_map[concept_1].keys()
    concept_2_sub_chapter = concept_sub_chapter_map[concept_2].keys()

    all_combination = list(itertools.product(concept_1_sub_chapter, concept_2_sub_chapter))

    result = []

    for i in all_combination:
        toc_1 = sub_chapters_toc[i[0]].split(".")
        toc_2 = sub_chapters_toc[i[1]].split(".")

        for index, j in enumerate(list(zip(toc_1, toc_2))):
            j_0 = int(j[0])
            j_1 = int(j[1])

            if j_0 != j_1:
                result.append((j_1 - j_0) / (2**index))
                break

    return sum(result) / len(result)


def indicator_function(p, q):
    if sub_chapters.index(p) < sub_chapters.index(q):
        return 1
    else:
        return -1


def alter_value(a):
    if a == 0:
        return 1
    if a == 1:
        return 0
