import React from 'react';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { shallow } from 'enzyme';
import SongGame from '../components/SongGame';

describe('SongGame Test', () => {
    it("should renders without crashing", () => {
        shallow(<SongGame />);
    });

});