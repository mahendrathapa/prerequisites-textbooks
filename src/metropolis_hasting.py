import numpy as np

from src import data
from src.utils import concept_concept_similarity, \
    concept_sub_chapter_similarity, calculate_complexity_level, indicator_function, alter_value

from src.config import ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4, ALPHA_5, BETA_1, BETA_2, BETA, EPOCHS, \
    PRINT_EPOCH


def cal_key_concept_extraction(concepts, sub_chapters, concept_sub_chapter_map):

    first_term = 0
    second_term = 0

    for i in concepts:
        for j in sub_chapters:
            first_term += concept_sub_chapter_map[i][j] * concept_sub_chapter_similarity(i, j)

    for i in concepts:
        for j in concepts:
            for p in sub_chapters:
                for q in sub_chapters:
                    if p != q:
                        second_term += concept_sub_chapter_map[i][p] * \
                            concept_sub_chapter_map[j][q] * concept_concept_similarity(i, j)

    total = ALPHA_1 * first_term - ALPHA_2 * second_term

    return total


def cal_pre_requisite_relationship(concepts, concept_map, concept_sub_chapter_map):

    first_term = 0
    second_term = 0

    for i in concepts:
        for j in concepts:
            if i != j:
                first_term += concept_map[i][j] * concept_concept_similarity(i, j)

    for i in concepts:
        for j in concepts:
            if i != j:
                second_term += concept_map[i][j] * calculate_complexity_level(i, j, concept_sub_chapter_map)

    total = ALPHA_3 * first_term + ALPHA_4 * second_term

    return total


def cal_joint_modeling(concepts, sub_chapters, concept_map, concept_sub_chapter_map):

    total = 0

    for i in concepts:
        for j in concepts:
            if i != j:
                for p in sub_chapters:
                    for q in sub_chapters:
                        total += indicator_function(p, q) * \
                            concept_sub_chapter_map[i][p] * concept_sub_chapter_map[j][q] * concept_map[i][j]

    return ALPHA_5 * total


def cal_CS_regularizer(concepts, sub_chapters, concept_sub_chapter_map):

    result = 0

    for i in concepts:
        for j in sub_chapters:
            result += concept_sub_chapter_map[i][j]

    return result


def cal_R_regularizer(concepts, concept_map):

    result = 0

    for i in concepts:
        for j in concepts:
            if i != j:
                result += concept_map[i][j]

    return result


def objective(concept_map, concept_sub_chapter_map):

    concepts = data.wikipedia_concepts
    sub_chapters = data.sub_chapters

    P_1 = cal_key_concept_extraction(concepts, sub_chapters, concept_sub_chapter_map)
    P_2 = cal_pre_requisite_relationship(concepts, concept_map, concept_sub_chapter_map)
    P_3 = cal_joint_modeling(concepts, sub_chapters, concept_map, concept_sub_chapter_map)
    R_1 = cal_CS_regularizer(concepts, sub_chapters, concept_sub_chapter_map)
    R_2 = cal_R_regularizer(concepts, concept_map)

    result = P_1 + P_2 + P_3 + BETA_1 * R_1 + BETA_2 * R_2
    return result


def take_decision(original_val, original_objective_val, alter_val, alter_objective_val):

    if original_objective_val <= alter_objective_val:
        transition_prob = 1.0
    else:
        transition_prob = np.exp(-BETA * (original_objective_val - alter_objective_val))

    rand_number = np.random.uniform(0, 1)

    if rand_number <= transition_prob:
        final_val = alter_val
        obj_val = alter_objective_val
    else:
        final_val = original_val
        obj_val = original_objective_val

    return final_val, obj_val


def metropolis_hasting(concept_map, concept_sub_chapter_map):

    concepts = data.wikipedia_concepts
    sub_chapters = data.sub_chapters

    for epoch in range(1, EPOCHS + 1):

        total_cs = 0

        for concept in concepts:
            for sub_chapter in sub_chapters:
                original_val = concept_sub_chapter_map[concept][sub_chapter]
                original_objective_val = objective(concept_map, concept_sub_chapter_map)

                alter_val = alter_value(original_val)
                concept_sub_chapter_map[concept][sub_chapter] = alter_val
                alter_objective_val = objective(concept_map, concept_sub_chapter_map)

                final_val, obj_val = take_decision(original_val,
                                                   original_objective_val,
                                                   alter_val,
                                                   alter_objective_val)

                total_cs += obj_val
                concept_sub_chapter_map[concept][sub_chapter] = final_val

        total_r = 0

        for i in concepts:
            for j in concepts:
                if i != j:
                    original_val = concept_map[i][j]
                    original_objective_val = objective(concept_map, concept_sub_chapter_map)

                    alter_val = alter_value(original_val)
                    concept_map[i][j] = alter_val
                    alter_objective_val = objective(concept_map, concept_sub_chapter_map)

                    final_val, obj_val = take_decision(original_val,
                                                       original_objective_val,
                                                       alter_val,
                                                       alter_objective_val)

                    total_r += obj_val
                    concept_map[i][j] = final_val

        if epoch % PRINT_EPOCH == 0:
            print(f"Epoch: {epoch} Objective value Concept Subchapter Map: {total_cs} " +
                  f"Objective value Concept Concept Map: {total_r}")
