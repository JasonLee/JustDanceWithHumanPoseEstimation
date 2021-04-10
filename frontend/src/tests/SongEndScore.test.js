import React from 'react';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { shallow } from 'enzyme';
import SongEndScore from '../components/SongEndScore';

jest.mock('react-router-dom', () => ({
    ...jest.requireActual('react-router-dom'), // use actual for all non-hook parts
    useParams: () => ({
      gameID: 'gameID',
    }),
    useRouteMatch: () => ({ url: '/results/gameID' }),
  }));


describe('SongEndScore Test', () => {
    it("should renders without crashing", () => {
        shallow(<SongEndScore />);
    });

    // TODO: finish once CSS is implemented
    it("should render ", () => {
        const wrapper = shallow(<SongEndScore />);
    });
        
});