import pprint

import numpy as np

from prerequisites_textbooks import data
from prerequisites_textbooks.metropolis_hasting import metropolis_hasting


np.random.seed(0)
pp = pprint.PrettyPrinter()


def generate_concept_map(concepts):

    result = {}

    for i in concepts:
        result[i] = {}
        for j in concepts:
            if i != j:
                result[i][j] = 0

    return result


def generate_concept_subchapter(concepts, sub_chapters):
    result = {}

    for i in concepts:
        result[i] = {}
        for j in sub_chapters:
            result[i][j] = 0
    return result


def main():
    concept_map = generate_concept_map(data.wikipedia_concepts)
    concept_sub_chapter_map = generate_concept_subchapter(data.wikipedia_concepts, data.sub_chapters)

    metropolis_hasting(concept_map, concept_sub_chapter_map)

    print("\nConcept Map\n")
    pp.pprint(concept_map)

    print("\n-------------------------------------\n")
    print("Concept Sub Chapter Map\n")
    pp.pprint(concept_sub_chapter_map)


if __name__ == "__main__":
    main()
