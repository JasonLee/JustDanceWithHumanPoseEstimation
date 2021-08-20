const request = require("supertest");
const app = require("./server");
const mongoose = require('mongoose');
require('dotenv').config();

describe("Test /songs route", () => {
    let server;
    let db;

    beforeAll(async (done) => {
        const connectionString = "mongodb+srv://" + process.env.db_user + ":" + process.env.db_pass + "@cluster0.tjbly.mongodb.net/" + process.env.db_name + "?retryWrites=true&w=majority";
        await mongoose.connect(connectionString, { useNewUrlParser: true, useUnifiedTopology: true });
        db = await mongoose.connection;

        // https://github.com/visionmedia/supertest/issues/520
        server = app.listen(4000, () => {
            global.agent = request.agent(server);
            done();
        });
    });

    xit("test getting song list", async () => {
        let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3QiLCJkaXNwbGF5TmFtZSI6IlBsYXllciIsImlkIjoxLCJpYXQiOjE2MTk3NTI2ODl9.2-dMn6pRt1c1RUaMEzm4kjUXUywIvpkMUvnRxEwE504";
        await global.agent.get("/songs", {
                headers: {
                    "Authorization": `Bearer ${token}`
                }
            })
            .then(response => {
                const firstEntry = {
                    _id: 1,
                    name: 'Feel Special',
                    artist: 'TWICE',
                    length: '1:30',
                    creator: 'Joe',
                    difficulty: 'Medium'
                };
        
                const result = response.body;
                let firstResult = result[0];
        
                // expect(firstResult).toEqual(firstEntry);
                expect(response.statusCode).toBe(200);
            });
    });

    it('test getting directly from db', async () => {
        let songsCollection = db.collection('songs');

        const firstEntry = {
            _id: 1,
            name: 'How You Like That',
            artist: 'Blackpink',
            length: '1:30',
            creator: 'Joe',
            difficulty: 'Medium'
        };

        const result = await songsCollection.find().toArray();
        var firstResult = result[0];

        expect(firstResult).toEqual(firstEntry);
    });

    it('test login happy path', async () => {
        await global.agent.post("/login")
            .send({
                username: 'jasonlee',
                password: 'Password1'
            }).then(response => {
                expect(response.body.token).toBeDefined();
                expect(response.statusCode).toBe(200);
            });
    });

    it('test login unhappy path', async () => {
        await global.agent.post("/login")
            .send({
                username: 'wronguser',
                password: 'Password1'
            }).then(response => {
                expect(response.body.token).toBeUndefined();
                expect(response.statusCode).toBe(401);
            });
    });

    it('test register happy path', async () => {
        await global.agent.post("/register")
            .send({
                username: 'testuser',
                password: 'Password1',
                email: 'testuser@test.com'
            }).then(response => {
                expect(response.text).toBe("User has been created");
                expect(response.statusCode).toBe(201);
            });
    });

    it('test register duplicate username unhappy path', async () => {
        await global.agent.post("/register")
            .send({
                username: 'user',
                password: 'Password1',
                email: 'duplicate@test.com'
            }).then(response => {
                expect(response.text).toBe("Error: Unable to create account");
                expect(response.statusCode).toBe(409);
            });
    });

    xit('test register and login happy path', async () => {
        await global.agent.post("/register")
            .send({
                username: 'testuserrl',
                password: 'Password1',
                email: 'reglog@test.com'
            });

        await global.agent.post("/login")
            .send({
                username: 'testuserrl',
                password: 'Password1'
            }).then(response => {
                expect(response.body.token).toBeDefined();
                expect(response.statusCode).toBe(200);
            });
    }); 

    afterAll(async () => {
        // Clean up test entries
        let usersCollection = db.collection('users');

        await usersCollection.deleteOne({username: 'testuser'});
        await usersCollection.deleteOne({username: 'testuserrl'});

        await server.close();
        await mongoose.disconnect();
    });

});