import React from 'react';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { shallow } from 'enzyme';
import ProfileBox from '../components/ProfileBox';

describe('ProfileBox Test', () => {
    it("should renders without crashing", () => {

        shallow(<ProfileBox />);
    });
});