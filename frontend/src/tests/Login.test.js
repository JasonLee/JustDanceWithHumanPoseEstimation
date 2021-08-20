import React from 'react';

import Enzyme, { shallow }  from 'enzyme';
import Login from '../components/Login';

import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { Router } from 'react-router-dom';
import { createMemoryHistory } from 'history';

import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
Enzyme.configure({ adapter: new Adapter() });

const mockAdapter = new MockAdapter(axios);
// Mocking Axios API request. Need to return promise to check state change.
mockAdapter.onPost("/login").reply(() => {
    return new Promise(function (resolve) {
        resolve([200, "TOKEN"]);
        reject(new Error("Should never error"));
    });
});

const flushPromises = () => new Promise(resolve => setImmediate(resolve));

// TODO: Write tests once integrated with everything
fdescribe('Login Test', () => {
    let token = "";

    beforeAll(async () => {
        
        let res = await axios.post("/login", {
            username: "test",
            password: "1234"
        });
        
        token = res.data
    });

    it("should renders without crashing", () => {
        const history = createMemoryHistory();
        shallow(<Router history={history}>
                    <Login setToken={()=>{}}/>
                </Router>);
    });

    it("should return a token", () => {
        const testGetToken = (token) =>  {
            expect(token).toBeDefined();
            expect(token).toBe("TOKEN");
        }

        const history = createMemoryHistory();
        shallow(<Router history={history}>
                    <Login setToken={testGetToken}/>
                </Router>);

    });

});