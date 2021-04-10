import React from 'react';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { shallow } from 'enzyme';
import ScoreCanvas, { getColorFromScore } from '../components/ScoreCanvas';
import { Stage, Layer, Line, Circle } from 'react-konva';

describe('SongCard Test', () => {
    it("should render without crashing", () => {
        const joints = [[1,1]];
        const mapping = [0];
        const results = [500];

        shallow(<ScoreCanvas joints={joints} mapping={mapping} results={results}/>);
    });

    it("should render the image with lines and circles", () => {
        const joints = [[1,1]];
        const mapping = [0];
        const results = [500];

        const wrapper = shallow(<ScoreCanvas joints={joints} mapping={mapping} results={results}/>);
        expect(wrapper.find(Stage).exists()).toBe(true);
        expect(wrapper.find(Layer).exists()).toBe(true);
        expect(wrapper.find(Line).exists()).toBe(true);
        expect(wrapper.find(Circle).exists()).toBe(true);
    });

    it("should return the correct color for score", () => {
        expect(getColorFromScore(500)).toBe("green");
        expect(getColorFromScore(300)).toBe("blue");
        expect(getColorFromScore(100)).toBe("yellow");
        expect(getColorFromScore(0)).toBe("red");

    });
});