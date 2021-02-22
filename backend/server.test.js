const request = require("supertest");
const app = require("./server");
const mongoose = require('mongoose');
require('dotenv').config();

describe("Test /songs route", () => {
    let db;

    beforeAll(async done => {
        const connectionString = "mongodb+srv://" + process.env.db_user + ":" + process.env.db_pass + "@cluster0.tjbly.mongodb.net/" + process.env.db_name + "?retryWrites=true&w=majority";
        await mongoose.connect(connectionString, { useNewUrlParser: true, useUnifiedTopology: true });
        db = await mongoose.connection;
        done();
    });

    test("MongoDB song test", async () => {
        return request(app)
            .get("/songs")
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
                var firstResult = result[0];
        
                expect(firstResult).toEqual(firstEntry);
                expect(response.statusCode).toBe(200);
            });
    });

    it('test getting directly from db', async () => {
        const songCollection = db.collection('songs')
        
        const firstEntry = {
            _id: 1,
            name: 'Feel Special',
            artist: 'TWICE',
            length: '1:30',
            creator: 'Joe',
            difficulty: 'Medium'
        };

        const result = await songCollection.find().toArray();
        var firstResult = result[0];

        expect(firstResult).toEqual(firstEntry);
    });

    afterAll(async () => {
        await db.close();
    });

});