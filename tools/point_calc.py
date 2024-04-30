from typing import Tuple


def point_calc_regular(math_study: Tuple[float, float],
                       hungarian_study: Tuple[float, float],
                       history_study: Tuple[float, float],
                       language_study: Tuple[float, float],
                       chosen_study: Tuple[float, float],
                       math_final: Tuple[float, bool],
                       hungarian_final: Tuple[float, bool],
                       history_final: Tuple[float, bool],
                       language_final: Tuple[float, bool],
                       chosen_final: Tuple[float, bool],
                       language_exam: None | str,
                       oktv_relevant: None | int,
                       oktv_irrelevant: None | int,
                       ) -> dict:
    """
    Calculates the university acceptance points for ELTE IK PTI BSc according to the regular point calculation rules.

    :param math_study: The end term grades from the last two studied years in high school in math in an
    ordered pair.
    :param hungarian_study: The end term grades from the last two studied years in high school in literature and grammar
    averaged in an ordered pair.
    :param history_study: The end term grades from the last two studied years in high school in history in an
    ordered pair.
    :param language_study: The end term grades from the last two studied years in high school in foreign language in an
    ordered pair.
    :param chosen_study: The end term grades from the last two studied years in high school in the chosen subject in
    an ordered pair.
    :param math_final: The score of the math final exam in percents, and a boolean which is True if the math final exam
    is an elevated level exam in an ordered pair.
    :param hungarian_final: The score of the hungarian final exam in percents, and a boolean which is True if the
    hungarian final exam is an elevated level exam in an ordered pair.
    :param history_final: The score of the history final exam in percents, and a boolean which is True if the history
    final exam is an elevated level exam in an ordered pair.
    :param language_final: The score of the language final exam in percents, and a boolean which is True if the language
    final exam is an elevated level exam in an ordered pair.
    :param chosen_final: The score of the chosen subject final exam in percents, and a boolean which is True if the
    chosen subject final exam is an elevated level exam in an ordered pair.
    :param language_exam: Either None if the student doesn't have a language exam or "B2" or "C1" as strings indicating
    the level of language exam that they have.
    :param oktv_relevant: None if the student doesn't have an oktv result from math or the chosen subject, or the
    student's placement in the competition if they have a result.
    :param oktv_irrelevant: None if the student doesn't have an oktv result from any subject other than math or the
    chosen subject, or the student's placement in the competition if they have a result.

    :returns: The calculated acceptance points.
    """
    study_points = (math_study[0] + math_study[1] + hungarian_study[0] + hungarian_study[1] + history_study[0] +
                    history_study[1] + language_study[0] + language_study[1] + chosen_study[0] + chosen_study[1]) * 2
    final_points_1 = (math_final[0] + hungarian_final[0] + history_final[0] + language_final[0] + chosen_final[0]) / 5

    final_points_2 = math_final[0] * (1 if math_final[1] else .67) + chosen_final[0] * (1 if chosen_final[1] else .67)

    institution_points = _calc_institution_points(math_final, chosen_final, language_exam, oktv_relevant,
                                                  oktv_irrelevant)

    return {"expected_points": (round(study_points) + round(final_points_1) +
                                round(final_points_2) + institution_points)}


def point_calc_double(math_final: Tuple[float, bool],
                      chosen_final: Tuple[float, bool],
                      language_exam: None | str,
                      oktv_relevant: None | int,
                      oktv_irrelevant: None | int) -> dict:
    """
    Calculates the university acceptance points for ELTE IK PTI BSc according to the double final exam point calculation
    rules.

    :param math_final: The score of the math final exam in percents, and a boolean which is True if the math final exam
    is an elevated level exam in an ordered pair.
    :param chosen_final: The score of the chosen subject final exam in percents, and a boolean which is True if the
    chosen subject final exam is an elevated level exam in an ordered pair.
    :param language_exam: Either None if the student doesn't have a language exam or "B2" or "C1" as strings indicating
    the level of language exam that they have.
    :param oktv_relevant: None if the student doesn't have an oktv result from math or the chosen subject, or the
    student's placement in the competition if they have a result.
    :param oktv_irrelevant: None if the student doesn't have an oktv result from any subject other than math or the
    chosen subject, or the student's placement in the competition if they have a result.
    
    :returns: The calculated acceptance points.
    """
    final_points_2 = math_final[0] * (1 if math_final[1] else .67) + chosen_final[0] * (1 if chosen_final[1] else .67)
    institution_points = _calc_institution_points(math_final, chosen_final, language_exam, oktv_relevant,
                                                  oktv_irrelevant)
    return {"expected_points": round(final_points_2) * 2 + institution_points}


def _calc_institution_points(math_final: Tuple[float, bool],
                             chosen_final: Tuple[float, bool],
                             language_exam: None | str,
                             oktv_relevant: None | int,
                             oktv_irrelevant: None | int):
    if oktv_relevant == "None":
        oktv_relevant = None
    if oktv_irrelevant == "None":
        oktv_irrelevant = None

    institution_points = 0
    if math_final[1] and math_final[0] >= 45:
        institution_points = institution_points + 50

    if chosen_final[1] and chosen_final[0] >= 45:
        institution_points = institution_points + 50

    if language_exam == 'B2':
        institution_points = institution_points + 28
    elif language_exam == 'C1':
        institution_points = institution_points + 50

    if oktv_relevant is not None and oktv_relevant <= 10:
        institution_points = institution_points + 100
    elif oktv_relevant is not None and oktv_relevant <= 20:
        institution_points = institution_points + 50
    elif oktv_relevant is not None and oktv_relevant <= 30:
        institution_points = institution_points + 25

    if oktv_irrelevant is not None and oktv_irrelevant <= 10:
        institution_points = institution_points + 20

    return min(100, institution_points)
