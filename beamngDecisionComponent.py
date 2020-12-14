#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

BEAMNG_DEFAULT_IMAGE_CLI = './beamngDecisionComponentDefaultImage.png'

FloatQuintuple = Tuple[float, float, float, float, float]


def euclidean_distance(vector: List[float]) -> float:
    return sum(map(lambda a: a**2, vector))**.5


class BeamNgDecisionComponent:
    def __init__(self, center_weight: float = 5, hint_weight: float = 4, danger_weight: float = 3, short_termness: float = 0.3):
        self.center_weight = center_weight
        self.hint_weight = hint_weight
        self.danger_weight = danger_weight
        self.short_termness = short_termness

    def linearize(self, frame: np.ndarray) -> np.ndarray:
        '''
        Any image that goes through here becomes binnary Black/White
        '''
        img = np.asarray(Image.fromarray(frame).convert('L'))
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        return img

    def analyze(self, frame: np.ndarray) -> Tuple[FloatQuintuple, FloatQuintuple, FloatQuintuple]:
        '''
        Transforms an image into a small segments matrix to be used as decision source
        [       0     1     2     3     4
            [ lt_d- lt_h- lt_c~ lt_h+ lt_d+ ],   0
            [ mt_d- mt_h- mt_c~ mt_h+ mt_d+ ],   1
            [ st_d- st_h- st_c~ st_h+ st_d+ ],   2
        ]
        lt_ => Long term
        mt_ => Medium term
        st_ => Short term
        d => Danger - demands  a quick  turn towards a   direction to avoid crashing
        h => Hint -   suggests a gentle turn towards a   direction to keep in the track
        c => Center - ignores  a any    turn towards any direction (special case for starting line)
        ~ => Neutralizes effect other suggestions
        - => Suggests a turn to the right (threat on the left)
        + => Suggests a turn to the left  (threat on the right)
        '''
        linear = self.linearize(frame)
        shape_y, shape_x = linear.shape
        segment_y_size = shape_y // 3
        segment_x_size = shape_x // 5
        shape = [[0]*5 for _ in range(3)]
        for segment_x_index in range(5):
            segment_x_stt = segment_x_size * (segment_x_index + 0)
            segment_x_end = segment_x_size * (segment_x_index + 1)
            for segment_y_index in range(3):
                segment_y_stt = segment_y_size * (segment_y_index + 0)
                segment_y_end = segment_y_size * (segment_y_index + 1)
                block = linear[segment_y_stt:segment_y_end, segment_x_stt:segment_x_end]
                shape[segment_y_index][segment_x_index] = block.mean() / 255
        return tuple([tuple(x) for x in shape])

    def decide(self, frame: np.ndarray) -> float:
        analysis = self.analyze(frame)
        analysis_mutable = [[x for x in y] for y in analysis]
        relevance_coeffs = [self.danger_weight,
                            self.hint_weight,
                            self.center_weight,
                            self.hint_weight,
                            self.danger_weight]
        for iy in range(3):
            importance_coeff = (1-self.short_termness)**(2-iy)
            for ix in range(5):
                analysis_mutable[iy][ix] = max(0,
                                               analysis_mutable[iy][ix]
                                               * relevance_coeffs[ix]
                                               * importance_coeff
                                               )
        turn_suggestion = [0, 0]
        no_turn_suggestion = 0
        for iy in range(3):
            for ix in range(2):
                side_suggestion = 0
                side_suggestion += analysis_mutable[iy][(3*ix)+0]
                side_suggestion += analysis_mutable[iy][(3 * ix) + 1]
                turn_suggestion[ix] += max(0, side_suggestion - analysis_mutable[iy][2])
                no_turn_suggestion += analysis_mutable[iy][2]
        dist = euclidean_distance(turn_suggestion + [no_turn_suggestion])
        turn_suggestion[0] *= -1
        turn_suggestion[0] /= dist
        turn_suggestion[1] /= dist
        return sum(turn_suggestion)


def main():
    decision_component = BeamNgDecisionComponent()
    frame = np.asarray(Image.open(BEAMNG_DEFAULT_IMAGE_CLI))
    print(decision_component.analyze(frame))
    print(decision_component.decide(frame))


if __name__ == "__main__":
    main()
