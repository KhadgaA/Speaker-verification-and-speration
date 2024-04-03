from itertools import accumulate
from functools import partial


def eer(labels, scores, verbose=False):
    """
    Args:
        labels: (N,1) with value being 0 or 1
        scores: (N,1) within [-1, 1]

    Returns:
        equal_error_rates
        threshold
    """
    joints = sorted(zip(scores, labels), key=lambda x: x[0])

    sorted_scores, sorted_labels = zip(*joints)

    total_ones = sum(sorted_labels)
    total_zeros = len(sorted_labels) - total_ones

    prefsum_ones = list(accumulate(sorted_labels,
                                   partial(_count_labels, label_to_count=1),
                                   initial=0))
    prefsum_zeros = list(accumulate(sorted_labels,
                                    partial(_count_labels, label_to_count=0),
                                    initial=0))

    ext_scores = [-1.0, *sorted_scores, 1.0]

    left, right = 0, len(ext_scores)

    for i in range(len(sorted_scores)):

        assert left <= right

        prev_left, prev_right = left, right

        idx = (left + right) // 2
        thresh = (ext_scores[idx] + ext_scores[idx+1]) / 2

        nb_positives = len(sorted_scores) - idx
        nb_negatives = idx

        nb_false_positives = total_zeros - prefsum_zeros[idx]
        nb_false_negatives = prefsum_ones[idx]

        false_positive_rate = nb_false_positives / nb_positives
        false_negative_rate = nb_false_negatives / nb_negatives

        if verbose:
            print(f"Round {i+1}")
            print(f"  => threshold = {thresh}")
            print(f"  => false positive rate = {false_positive_rate}")
            print(f"  => false negative rate = {false_negative_rate}")

        if false_positive_rate > false_negative_rate:
            left = idx
        elif false_positive_rate < false_negative_rate:
            right = idx
        else:
            break

        if prev_left == left and prev_right == right:
            break

    equal_error_rate = (false_positive_rate + false_negative_rate) / 2

    return equal_error_rate, thresh


def _count_labels(counted_so_far, label, label_to_count=0):
    return counted_so_far + 1 if label == label_to_count else counted_so_far


if __name__ == "__main__":
    test_labels = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1]
    test_scores = [0.2, 0.7, -0.3, 0.5, 0.4, 0.9, 0.1, 0.3, 0.45, 0.6, 0.25]
    print("equal error rate = {}, threshold = {}".format(
        *eer(test_labels, test_scores)))