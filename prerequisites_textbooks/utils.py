def concept_concept_similarity(concept_1, concept_2):
    return 1.0


def concept_sub_chapter_similarity(concept, sub_chapter):
    return 1.0


def calculate_complexity_level(concept):
    return 1.0


def indicator_function(p, q):
    if p < q:
        return 1
    else:
        return -1


def alter_value(a):
    if a == 0:
        return 1
    if a == 1:
        return 0
