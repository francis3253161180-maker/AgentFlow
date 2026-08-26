from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    groundtruth: str
    answer: str
    expected: bool
    expected_route: str


# These are independently authored discourse cases, not transformations of
# the 40 saved mini20 rollouts.
SYNTHETIC_CASES = [
    SyntheticCase("date_corrected_to_wrong", "June 14, 2021", "June 14, 2021 is incorrect; the actual date was September 2, 2022.", False, "judge"),
    SyntheticCase("date_wrong_corrected_to_right", "September 2, 2022", "I first said June 14, 2021, but the actual date is September 2, 2022.", True, "judge"),
    SyntheticCase("yes_to_no_self_correction", "Yes", "Yes was my initial thought, but no—the claim is false.", False, "judge"),
    SyntheticCase("no_to_yes_self_correction", "Yes", "No at first; after checking the evidence, actually yes.", True, "judge"),
    SyntheticCase("date_mentioned_then_denied", "March 11, 2020", "The date was March 11, 2020? No, it was July 8, 2020.", False, "judge"),
    SyntheticCase("entity_near_but_rejected", "Lake Victoria", "The route passes near Lake Victoria, but the destination is elsewhere.", False, "judge"),
    SyntheticCase("multiple_candidate_entities", "Eastport", "It could be Northport or Eastport; after checking, Eastport is the answer.", True, "judge"),
    SyntheticCase("near_but_not_in_entity", "Grant Park", "The gallery is near Grant Park, but not in Grant Park.", False, "judge"),
    SyntheticCase("thought_x_actually_y", "Port Royal", "I thought the harbor was Kingston; actually it was Port Royal.", True, "judge"),
    SyntheticCase("final_marker_overrides_earlier_reasoning", "Milan", "I first considered Turin, but the evidence supports this conclusion. Final Answer: Milan", True, "deterministic"),
    SyntheticCase("final_xml_answer_overrides_candidate", "Eastport", "Northport was considered during reasoning. <answer>Eastport</answer>", True, "deterministic"),
    SyntheticCase("fraction_local_proof", r"\dfrac{3}{8}", r"\frac{3}{8}", True, "deterministic"),
    SyntheticCase("radical_local_equivalence", r"2 + \sqrt{3}", "√3 + 2", True, "deterministic"),
    SyntheticCase("integer_local_match", "47", "The final answer is 47.", True, "deterministic"),
    SyntheticCase("integer_local_mismatch", "47", "The final answer is 52.", False, "deterministic"),
    SyntheticCase("date_local_match", "November 5, 2019", "<answer>November 5, 2019</answer>", True, "deterministic"),
    SyntheticCase("bare_no_local", "No", "No.", True, "deterministic"),
    SyntheticCase("bare_yes_local", "Yes", "<answer>Yes</answer>", True, "deterministic"),
    SyntheticCase("entity_mentioned_then_rejected", "Alexandria", "Alexandria is mentioned in the report, but the actual site is Memphis.", False, "judge"),
    SyntheticCase("long_correct_entity_explanation", "Ada Lovelace", "The evidence describes the early programmer who worked on the analytical engine; the person identified is Ada Lovelace.", True, "judge"),
    SyntheticCase("correct_phrase_then_wrong_replacement", "River Thames", "River Thames was my first answer, but the final river is the Seine.", False, "judge"),
    SyntheticCase("wrong_phrase_then_correct_replacement", "Seine", "I initially selected the River Thames; after correction, the answer is Seine.", True, "deterministic"),
    SyntheticCase("date_wrong_candidate_only", "December 1, 2020", "The event took place on August 9, 2020.", False, "deterministic"),
    SyntheticCase("yes_no_contradiction_with_final_no", "No", "Yes was the first thought. Final answer: No.", True, "deterministic"),
    SyntheticCase("entity_rejection_with_explicit_final", "Northport", "Northport is not correct. Final Answer: Southport", False, "judge"),
    SyntheticCase("target_date_in_quote_then_new_date", "February 2, 2022", "The note quotes February 2, 2022, but says the verified date is October 17, 2023.", False, "judge"),
]
