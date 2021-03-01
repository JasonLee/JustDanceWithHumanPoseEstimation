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

    it("test getting song list", async () => {
        await global.agent.get("/songs")
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
        
                expect(firstResult).toEqual(firstEntry);
                expect(response.statusCode).toBe(200);
            });
    });

    it('test getting directly from db', async () => {
        let songsCollection = db.collection('songs');

        const firstEntry = {
            _id: 1,
            name: 'Feel Special',
            artist: 'TWICE',
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
                username: 'user',
                password: '1234'
            }).then(response => {
                expect(response.body.token).toBeDefined();
                expect(response.statusCode).toBe(200);
            });
    });

    it('test login unhappy path', async () => {
        await global.agent.post("/login")
            .send({
                username: 'wronguser',
                password: '1234'
            }).then(response => {
                expect(response.body.token).toBeUndefined();
                expect(response.statusCode).toBe(401);
            });
    });

    it('test register happy path', async () => {
        await global.agent.post("/register")
            .send({
                username: 'testuser',
                password: '1234',
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
                password: '1234',
                email: 'duplicate@test.com'
            }).then(response => {
                expect(response.text).toBe("Error: Unable to create acount");
                expect(response.statusCode).toBe(409);
            });
    });

    it('test register and login happy path', async () => {
        await global.agent.post("/register")
            .send({
                username: 'testuserrl',
                password: '1234',
                email: 'reglog@test.com'
            });

        await global.agent.post("/login")
            .send({
                username: 'testuserrl',
                password: '1234'
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