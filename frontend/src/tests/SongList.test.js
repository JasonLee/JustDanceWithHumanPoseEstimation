import React from 'react';
import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { mount, shallow } from 'enzyme';
import { ListItem } from '@material-ui/core/';

import Music from '../components/Music';
import SongCard from '../components/SongCard';
import SongList from '../components/SongList';

const mockAdapter = new MockAdapter(axios);
// Mocking Axios API request. Need to return promise to check state change.
mockAdapter.onGet("/songs").reply(() => {
    return new Promise(function (resolve) {
        resolve([200, [{ "_id": 1, "name": "song1" }, { "_id": 2, "name": "song2" }]]);
    });
});

const flushPromises = () => new Promise(resolve => setImmediate(resolve));

describe('SongList Test', () => {
    beforeAll(() => {
        // JSDOM does not support video and audio elements
        jest.spyOn(window.HTMLMediaElement.prototype, 'pause').mockImplementation(() => { })
        jest.spyOn(window.HTMLMediaElement.prototype, 'play').mockImplementation(() => { })
    });


    it("should renders without crashing", () => {
        shallow(<SongList />);
    });

    it("should renders song list", async () => {
        const wrapper = shallow(<SongList />);
        await flushPromises();
        // Renders new children components
        wrapper.update();
        expect(wrapper.find(ListItem)).toHaveLength(2);

    });

    it('should show song card when clicked', async () => {
        const wrapper = shallow(<SongList />);
        await flushPromises()
        wrapper.find(ListItem).first().simulate('click');

        expect(wrapper.find(SongCard).exists()).toBe(true);
        expect(wrapper.find(Music).exists()).toBe(true);
        expect(wrapper.find(ListItem).exists()).toBe(false);
    });

    // Can't simulate mouse click outside component
    it.skip('should show list when clicked off', () => {
        const wrapper = mount(<SongList />);

        wrapper.find(ListItem).first().simulate('click');

        // Outside click

        expect(wrapper.find(SongCard).exists()).toBe(false);
        expect(wrapper.find(Music).exists()).toBe(false);
        expect(wrapper.find(ListItem).exists()).toBe(true);


    });
});