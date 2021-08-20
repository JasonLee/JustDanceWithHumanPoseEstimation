import React from 'react';
import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';

import Enzyme from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

import { mount, shallow } from 'enzyme';
import { ListItem } from '@material-ui/core/';

import { Router } from 'react-router-dom';
import { createMemoryHistory } from 'history';

import Music from '../components/Music';
import SongCard from '../components/SongCard';
import SongList from '../components/SongList';
import Song from '../components/Song';

const mockAdapter = new MockAdapter(axios);
// Mocking Axios API request. Need to return promise to check state change.
mockAdapter.onGet("/songs").reply(() => {
    return new Promise(function (resolve) {
        resolve([200, [
            {"_id": 1, "name": "csong", "artist": "cartist"}, 
            {"_id": 2, "name": "bsong", "artist": "bartist"}, 
            {"_id": 3, "name": "asong", "artist": "aartist"}
        ]]);
        reject(new Error("Should never error"));
    });
});

const flushPromises = () => new Promise(resolve => setImmediate(resolve));

describe('SongList display and API Test', () => {
    let wrapper;

    beforeAll(() => {
        // JSDOM does not support video and audio elements
        jest.spyOn(window.HTMLMediaElement.prototype, 'pause').mockImplementation(() => { })
        jest.spyOn(window.HTMLMediaElement.prototype, 'play').mockImplementation(() => { })
    });

    beforeEach(async () => {
        wrapper = shallow(<SongList.WrappedComponent match={ { params: { lobbyID: "" } }} token={"TEST"} />);
        await flushPromises();
        // Renders new children components
        wrapper.update();
    });


    it("should renders without crashing", () => {
        shallow(<SongList token={"TEST TOKEN"}/>);
    });

    it("should renders song list", () => {
        expect(wrapper.find(Song)).toHaveLength(3);
    });

    // Material-ui giving issues with onClick testing
    it.skip('should show song card when clicked', () => {
        wrapper.find('WithStyles(ForwardRef(ListItem))').first().simulate('click');

        console.log(wrapper.debug());

        expect(wrapper.find(SongCard).exists()).toBe(true);
        expect(wrapper.find(Music).exists()).toBe(true);
        expect(wrapper.find(ListItem).exists()).toBe(false);
    });

    // Can't simulate mouse click outside component
    it.skip('should show list when clicked off', () => {
        wrapper.find(ListItem).first().simulate('click');

        // Outside click

        expect(wrapper.find(SongCard).exists()).toBe(false);
        expect(wrapper.find(Music).exists()).toBe(false);
        expect(wrapper.find(ListItem).exists()).toBe(true);


    });
});

describe('SongList sort and filter test', () => {
    let wrapper;
    
    beforeEach(async () => {
        wrapper = shallow(<SongList.WrappedComponent match={ { params: { lobbyID: "" } }} token={"TEST"} />);
        await flushPromises();
        // Renders new children components
        wrapper.update();
    });

    it("should renders without crashing", () => {
        shallow(<SongList token={"TEST TOKEN"}/>);
    });

    it("should filter songs when searching", async () => {
        const input = wrapper.find('input');
        input.simulate('change', { target: { value: 'csong' } });

        expect(wrapper.find(Song)).toHaveLength(1);
    });

    it("should return 0 when searching for non-existing song", async () => {
        const input = wrapper.find('input');
        input.simulate('change', { target: { value: 'dsong' } });

        expect(wrapper.find(Song)).toHaveLength(0);
    });

    it('should select filter by song name and sort properly', async () => {
        // Simulate selecting options
        const select = wrapper.find('WithStyles(ForwardRef(Select))');
        select.simulate('change', { target: { value: 'name' } });
        
        for (let i = 0; i < 2; i++) {
            expect(wrapper.find(Song).at(i).props().data.name < wrapper.find(Song).at(i+1).props().data.name).toBeTruthy();
        }
        
    });

    it('should select filter by song artist and sort properly', async () => {
         // Simulate selecting options
        const select = wrapper.find('WithStyles(ForwardRef(Select))');
        select.simulate('change', { target: { value: 'artist' } });
        
        for (let i = 0; i < 2; i++) {
            expect(wrapper.find(Song).at(i).props().data.artist < wrapper.find(Song).at(i+1).props().data.artist).toBeTruthy();
        }
    });

});