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
import Song from '../components/Song';

const mockAdapter = new MockAdapter(axios);
// Mocking Axios API request. Need to return promise to check state change.
mockAdapter.onGet("/songs").reply(() => {
    return new Promise(function (resolve) {
        resolve([200, [{ "_id": 1, "name": "song1" }, { "_id": 2, "name": "song2" }]]);
    });
});

const flushPromises = () => new Promise(resolve => setImmediate(resolve));

describe('SongList display and API Test', () => {
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
        expect(wrapper.find(Song)).toHaveLength(2);

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

describe('SongList sort and filter test', () => {
    const songList = [{"_id": 1, "name": "csong", "artist": "cartist"}, {"_id": 2, "name": "bsong", "artist": "bartist"}, {"_id": 3, "name": "asong", "artist": "aartist"}]

    it("should renders without crashing", () => {
        shallow(<SongList />);
    });

    it("should filter songs when searching", () => {
        const wrapper = shallow(<SongList />);
        wrapper.setState({songs: songList})

        const input = wrapper.find('input');
        input.simulate('change', { target: { value: 'csong' } });

        expect(wrapper.find(Song)).toHaveLength(1);
    });

    it("should return 0 when searching for non-existing song", () => {
        const wrapper = shallow(<SongList />);
        wrapper.setState({songs: songList})

        const input = wrapper.find('input');
        input.simulate('change', { target: { value: 'dsong' } });

        expect(wrapper.find(Song)).toHaveLength(0);
    });

    it('should select filter by song name and sort properly', () => {
        const wrapper = shallow(<SongList />);
        wrapper.setState({songs: songList})
        
        // Simulate selecting options
        const select = wrapper.find('select');
        select.simulate('change', { target: { value: 'name' } });
        
        for (let i = 0; i < songList.length-1; i++) {
            expect(wrapper.find(Song).at(i).props().data.name < wrapper.find(Song).at(i+1).props().data.name).toBeTruthy();
        }
        
    });

    it('should select filter by song artist and sort properly', () => {
        const wrapper = shallow(<SongList />);
        wrapper.setState({songs: songList})
        
        // Simulate selecting options
        const input = wrapper.find('select');
        input.simulate('change', { target: { value: 'artist' } });
        
        for (let i = 0; i < songList.length-1; i++) {
            expect(wrapper.find(Song).at(i).props().data.artist < wrapper.find(Song).at(i+1).props().data.artist).toBeTruthy();
        }
        
    });




});