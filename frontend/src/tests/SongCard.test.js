import React from 'react';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { shallow } from 'enzyme';
import SongCard from '../components/SongCard';

describe('SongCard Test', () => {
    it("should renders without crashing", () => {
        const data = {
            "name": "TestName",
            "artist": "TestArtist",
            "length": "1:30",
            "creator": "Joe",
            "difficulty": "Hard",
        }

        shallow(<SongCard data={data}/>);
    });

    it("should renders song card details", () => {
        const data = {
            "name": "TestName",
            "artist": "TestArtist",
            "length": "1:30",
            "creator": "Joe",
            "difficulty": "Hard",
        };

        const wrapper = shallow(<SongCard data={data}/>);
        expect(wrapper.contains("TestName")).toEqual(true);
        expect(wrapper.contains("TestArtist")).toEqual(true);
        expect(wrapper.contains("1:30")).toEqual(true);
        expect(wrapper.contains("Joe")).toEqual(true);
        expect(wrapper.contains("Hard")).toEqual(true);
    });
});