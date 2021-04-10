const mongoose = require('mongoose');
const express = require('express');
const queue = require('express-queue');
const passport = require('passport');
const BearerStrategy = require('passport-http-bearer').Strategy;
const ms = require('mediaserver');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const axios = require('axios');
const { response } = require('express');
const pose = require('./PoseComparision.js');
const math = require("mathjs")
const { v4: uuidv4 } = require('uuid');


require('dotenv').config();

const app = express();
app.use(express.json({limit: '50mb'}));
const port = 8000;
const saltRounds = 10;
const SCALE = 1000;
const connectionString = "mongodb+srv://" + process.env.db_user + ":" + process.env.db_pass + "@cluster0.tjbly.mongodb.net/" + process.env.db_name + "?retryWrites=true&w=majority";

mongoose.connect(connectionString, { useNewUrlParser: true, useUnifiedTopology: true });
const db = mongoose.connection;

db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', function () {
    console.log('Connected to Database');
});
const songCollection = db.collection('songs');
const userCollection = db.collection('users');
const gameCollection = db.collection('games');

passport.use(new BearerStrategy(
    function (token, done) {
        jwt.verify(token, process.env.private_key, function (err, decoded) {
            if (err) { return done(err) }
            if (decoded) {
                console.log(decoded.username)
                return done(null, decoded.username);
            } else {
                return done(null, false);
            }
        });
    }
));

// Test check user's bearer token
app.get('/', passport.authenticate('bearer', { session: false }),
    function (req, res) {
        res.status(200).send("access_token is OK");
    }
);

function generateToken(username) {
    const token = jwt.sign({ username: username }, process.env.private_key);
    return token
}

app.post('/register', (req, res) => {
    userCollection.findOne({ "username": req.body.username })
        .then(result => {
            if (result) {
                res.status(409).send("Error: Unable to create acount");
                // console.log("Username already exists");
                return;
            } else {
                // Not duplicate
                bcrypt.hash(req.body.password, saltRounds, function (err, hash) {
                    if (hash) {
                        // TODO: user db fields
                        userCollection.insertOne({
                            username: req.body.username,
                            password: hash,
                            email: req.body.email
                        }).then((acknowledged, insertedId) => {
                            if (acknowledged) {
                                res.status(201).send("User has been created");
                            }
                        }).catch(err => console.log(err));
                    }
                });
            }
        }).catch(err => {
            res.status(409).send(err);
            return;
        });
});

app.post('/login', (req, res) => {
    userCollection.findOne({ "username": req.body.username })
        .then(result => {
            if (result) {
                bcrypt.compare(req.body.password, result.password, (err, result) => {
                    if (err) { console.log(err) }
                    if (result) {
                        const token = generateToken(req.body.username);
                        res.status(200).send({
                            token: token
                        });
                    } else {
                        res.sendStatus(401);
                    }
                });
            } else {
                res.sendStatus(401);
            }
        }).catch(err => console.log(err));
});


const getPoseData = async (songName, timestamp) => {
    const response = await songCollection.aggregate([
        { $match: { "name": songName } },
        { $project: { _id: 0, data: 1 } },
        { $unwind: '$data' },
        { $match: { "data.timestamp": {$gte: timestamp} } },
        { $sort: { "timestamp": -1 } },
        { $limit: 1 }
    ], { "allowDiskUse" : true });

    return response.toArray()

}

const processImageToPose = async (base64Data, song, timestamp) => {
    const response = await axios.post("http://localhost:5000/pose", {
            image: base64Data,
            timestamp: timestamp
        },
        {
            headers: {
            'Content-Type': 'application/json',
            }
        });

    return response.data
}

// Middleware to check if a game is in session
const checkGameInPlay = () => {
    return (req, res, next) => {
        if (!req.body.gameID) {
            res.status(400).send('No game id was passed');
            return
        }

        gameCollection.findOne({ "gameID": req.body.gameID })
        .then(result => {
            if (result) {
                next()
            }else{
                res.status(400).send("Error: game does not exist")
                return
            }
        }).catch(err => res.status(400).send("An error has occured: " + err));

      }
}

app.post('/pose_score', checkGameInPlay(), async (req, res) => {
    console.log("SENDING IMAGE TO BACKEND");
    let timestamp = req.body.timestamp
    let songName = req.body.songName

    let base64Data = req.body.image.replace(/^data:image\/webp;base64,/, "");

    // console.log(req.body.image)
    // console.log("songName", songName)
    // console.log("timestamp", timestamp)
    const truthData = await getPoseData(songName, timestamp)
    const userData = await processImageToPose(base64Data, songName, timestamp)
    const user_joint_map = userData.points
    const truth_joint_map = truthData[0].data.joint_map

    results = pose.compare_poses(user_joint_map, truth_joint_map)

    user_joints_scaled = math.dotMultiply(userData.points, SCALE)
    truth_joints_scaled = math.dotMultiply(truth_joint_map, SCALE)

    console.log("truth_joints_scaled", truth_joints_scaled)
    console.log("user_joints_scaled", user_joints_scaled)
    
    totalScore = results.reduce((a, b) => a + b, 0)
    res.send({
        "joints": user_joints_scaled,
        "mapping": userData.mapping,
        "truth_joints": truth_joints_scaled,
        "results": results,
        "score": totalScore
    });

    increaseScore(req.body.gameID, totalScore);

    // Add to game record

    // res.status(200).send(userData);
    // // Webcam image to process
    // require("fs").writeFile("out.png", base64Data, 'base64', function(err) {
    //     res.sendStatus(200);
    // });
});

const increaseScore = (gameID, scoreInc) => {
    gameCollection.updateOne({
        gameID: gameID
    },{
        $inc: { score: scoreInc }
    });
}

// Client calls api route to create game in db
// Return gameID
app.post('/create_game', (req, res) => {
    const songID = req.body.songID;
    const gameID = uuidv4();
    const newDate = new Date();
    const datetime = newDate.toISOString()

    // Note MONGODB has UUID support
    gameCollection.insertOne({
        songID: songID,
        gameID: gameID,
        score: 0,
        accuracy: 0.00,
        active: true,
        created: {
            datetime: datetime
        }
    }).then((acknowledged, insertedId) => {
        // TODO: add expiry
        if (acknowledged) {
            res.status(201).send({
                "message": "Game has been created",
                "gameID": gameID
                
            });
        }
    }).catch(err => console.log(err));
});

app.post('/finish_game', (req, res) => {
    gameID = req.body.gameID;

    gameCollection.findOneAndUpdate({
        gameID: gameID
    },{
        $set: { active: false }
    }, (err, documents) => {
        res.send({ error: err, game: documents.value });
    });
});

app.get('/songs/:id', (req, res) => {
    let id = req.params.id;
    // ms.pipe(req, res, "./dance_cover.mp4");
    ms.pipe(req, res, "../ml_backend/video.mp4");

});


app.get('/songs', (req, res) => {
    songCollection.find().toArray()
        .then(results => {
            res.status(200).send(results);
        }).catch(error => console.error(error));
})

app.get('/audio', (req, res) => {
    ms.pipe(req, res, "../frontend/src/components/audio.mp3");
})

app.get('/audio2', (req, res) => {
    ms.pipe(req, res, "../frontend/src/components/audio2.mp3");
})

if (process.env.NODE_ENV != "test") {
    app.listen(port, () => {
        console.log(`Server listening on port:${port}`);
    });
}

module.exports = app;


