import React from 'react';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { shallow } from 'enzyme';
import Song from '../components/Song';

describe('Song Test', () => {
    it("should renders without crashing", () => {
        const data = {
            "name": "TestName",
            "artist": "TestArtist"
        }
        shallow(<Song data={data}/>);
    });

    it("should renders song details", () => {
        const data = {
            "name": "TestName",
            "artist": "TestArtist"
        }
        const wrapper = shallow(<Song data={data}/>);
        const name = "TestName";
        const artist = "TestArtist";
        expect(wrapper.contains(name)).toEqual(true);
        expect(wrapper.contains(artist)).toEqual(true);
    });
});