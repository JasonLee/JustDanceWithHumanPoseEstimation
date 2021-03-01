import React from 'react';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { shallow } from 'enzyme';
import Login from '../components/Login';

// TODO: Write tests once integrated with everything
describe('Login Test', () => {
    fit("should renders without crashing", () => {
        shallow(<Login setToken={()=>{}}/>);
    });

    xit("correct username and password", () => {
        testGetToken = (token) =>  {
            expect(token).toBeDefined();
        }
        const wrapper = shallow(<Login setToken={()=>{}}/>);

        const input = wrapper.find({ type: "text" });
        input.simulate('change', { target: { value: 'user' } });

    });

});