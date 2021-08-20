const mongoose = require('mongoose');
const express = require('express');
const queue = require('express-queue');
const passport = require('passport');
const BearerStrategy = require('passport-http-bearer').Strategy;
const ms = require('mediaserver');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const axios = require('axios');
const pose = require('./PoseComparision.js');
const sendEmail = require('./sendEmail');
const math = require("mathjs")
const { v4: uuidv4 } = require('uuid');

const http = require('http');

require('dotenv').config();

const app = express();
app.use(express.json({ limit: '50mb' }));

const server = http.createServer(app);
const socketIo = require("socket.io")(server);
const redisAdapter = require('socket.io-redis');
socketIo.adapter(redisAdapter({ host: 'localhost', port: 6379 }));

const port = 8000;

const swaggerUi = require('swagger-ui-express');
const swaggerDocument = require('./swagger.json');

// Stock image base64
const stockImage = "data:image/png;base64,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";

const saltRounds = 10;
const SCALE = 300;
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
const lobbyCollection = db.collection('lobbies');

// Authentication will use Bearer tokens with JWT
passport.use(new BearerStrategy(
    function (token, done) {
        jwt.verify(token, process.env.private_key, function (err, decoded) {
            if (err) { return done(err) }
            if (decoded) {
                return done(null, decoded);
            } else {
                return done(null, false);
            }
        });
    }
));

// Test check user's bearer token
app.get('/', passport.authenticate('bearer', { session: false }), (req, res) => {
    // req.user
    res.status(200).send("AUTH OK");
});

// SWAGGER DOCS
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// Generates a JWT token with user details as payload
function generateToken(data) {
    const token = jwt.sign({ username: data.username, id: data._id }, process.env.private_key, { expiresIn: '24h' });
    return token
}

const CLIENT_ORIGIN = "http://localhost:3000"

const confirm = (id) => ({
    subject: 'React Confirm Email',
    html: `
      <a href='${CLIENT_ORIGIN}/confirm/${id}'>
        click to confirm email
      </a>
    `,
    text: `Copy and paste this link: ${CLIENT_ORIGIN}/confirm/${id}`
});

const sendRegistrationEmail = (email, id) => {
    console.log("Sending verification email to ", email);
    sendEmail(email, confirm(id));
}



app.post('/verifyAccount', (req, res) => {
    const id = req.body.id;
    let ObjectId = require('mongodb').ObjectId; 

    userCollection.findOne({ _id: ObjectId(id) }).then(user => {
        // The user exists but has not been confirmed. We need to confirm this 
        // user and let them know their email address has been confirmed.
        if (user && !user.activated) {
            userCollection.updateOne({
                _id: ObjectId(id)
            },{
                $set: { activated: true }
            },(err, result) => {
                if (err) throw err;
                console.log(user.username, "- Email has been verified");
                res.json({ msg: "Email has been verified" })
            });
        }
        // The user has already confirmed this email address.
        else {
            console.log(user.username, "- Attempted to verify their verified email");
            res.json({ msg: "Email has already been verified" });
        }

    }).catch(err => console.error("Error: verifyAccount user:", user.username, err))

});

const validateDetails = (email, password) => {
    const passwordCheck = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[0-9a-zA-Z]{8,}$/;
    const emailCheck = /^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])/;
    
    return emailCheck.test(email) &&  passwordCheck.test(password);
}

// Registering - uses BCrypt to hash passwords
app.post('/register', (req, res) => {
    // Serverside validation
    if (!validateDetails(req.body.email, req.body.password)) {
        res.status(409).send("Error: Unable to create account");
        console.error("New user", req.body.username, "Invalid Email address provided");
    }

    userCollection.findOne({ "username": req.body.username })
        .then(result => {
            if (result) {
                res.status(409).send("Error: Unable to create account");
                console.log("Username already exists");
                return;
            } else {
                // Not duplicate
                bcrypt.hash(req.body.password, saltRounds, (err, hash) => {
                    if (hash) {
                        console.log("req.body.profilePic", req.body.profilePic);
                        // TODO: user db fields
                        userCollection.insertOne({
                            username: req.body.username,
                            password: hash,
                            email: req.body.email,
                            profilePic: req.body.profilePic === undefined || req.body.profilePic === null ? stockImage : req.body.profilePic,
                            created: new Date(),
                            activated: false
                        }, (err, result) => {
                            if (err) throw err;
                            console.log("User", req.body.username, "has successfully created an account", result);
                            sendRegistrationEmail(req.body.email, result.insertedId);
                            res.status(201).send("User has been created");
                        });
                    }
                });
            }
        }).catch(err => {
            res.status(409).send(err);
            console.error("/register an error has occurred when trying to add account to DB", err);
            return;
        });
});

// Login
app.post('/login', (req, res) => {
    userCollection.findOne({ "username": req.body.username, "activated": true })
        .then(result => {
            if (result) {
                bcrypt.compare(req.body.password, result.password, (err, valid) => {
                    if (err) { console.error(err) }
                    if (valid) {
                        const token = generateToken(result);
                        res.status(200).send({
                            token: token
                        });
                    } else {
                        console.error("LOGIN ATTEMPT FAILED - password mismatch",  req.body.username);
                        res.sendStatus(401);
                    }
                });
            } else {
                console.error("LOGIN ATTEMPT FAILED - username doesn't exist",  req.body.username);
                res.sendStatus(401);
            }
        }).catch(err => console.error("LOGIN ATTEMPT FAILED - DB Query issue",  req.body.username, err));
});

// Get user info for profile box
app.get('/getUserDisplayInfo', passport.authenticate('bearer', { session: false }), (req, res) => {
    const username = req.query.username || req.user.username;

    userCollection.findOne({ "username": username }, {
        fields: {
            _id: 0,
            username: 1,
            profilePic:1
        }
    }).then(result => {
        res.send(result)
    }).catch(err => console.error("/getUserDisplayInfo - DB Query issue",  username, err));

});

// Get User scores for song leaderboard
app.get('/getUserSongScores', passport.authenticate('bearer', { session: false }), (req, res) => {
    const playerID = req.user.id;
    const songID = parseInt(req.query.songID);

    // REFACTOR 
    gameCollection.find({
        "playerID": mongoose.Types.ObjectId(playerID),
        "songID": songID
    }, {
        fields: {
            _id: 0,
            gameID: 1,
            score: 1,
        }
    }).toArray().then(result => {
        res.send(result);
    }).catch(err => console.error("/getUserSongScores - DB Query issue. playerID",  playerID, "songID", songID , err));
});

// Get top scores from all users on a song for leaderboard
app.get('/getGlobalSongScores', passport.authenticate('bearer', { session: false }), (req, res) => {
    const songID = parseInt(req.query.songID);

    gameCollection.aggregate([
    {
        $lookup: {
          from: "users",
          localField: "playerID",
          foreignField: "_id",
          as: "user"
        }
    },{
        $unwind: "$user"
    }, {
        $match: {
            "songID": songID
        }
    },{
        $project: {
            _id: 0,
            
            gameID: 1,
            score: 1,
            "user.username": 1,
            "user.profilePic": 1
        }
    },{ 
        $sort: {
            score: -1
        }
    },{ 
        $limit: 20
    }]).toArray().then(result => {
        res.send(result)
    }).catch(err => console.error("/getGlobalSongScores - DB Query issue. songID", songID , err));
});

// Get pose data from DB at timestamp to be used for pose comparision
const getPoseData = async (songID, timestamp) => {
    // Query to search within song to get pose data of a timestamp
    // Requires unwinding arrays
    const response = await songCollection.aggregate([
        { $match: { "_id": songID } },
        { $project: { _id: 0, data: 1 } },
        { $unwind: '$data' },
        { $match: { "data.timestamp": { $gte: timestamp } } },
        { $sort: { "timestamp": -1 } },
        { $limit: 1 }
    ], { allowDiskUse: true });

    return response.toArray();
}

// Gets pose estimation via API call to ML server
const processImageToPose = async (base64Data, timestamp) => {
    const response = await axios.post("http://localhost:5000/pose", {
        image: base64Data,
        timestamp: timestamp
    },{
        headers: {
            'Content-Type': 'application/json',
        }
    });

    return response.data;
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
                } else {
                    res.status(400).send("Error: game does not exist")
                    return
                }
            }).catch(err => res.status(400).send("An error has occured: " + err));

    }
}

// API call from frontend to process image to get scores
app.post('/pose_score', passport.authenticate('bearer', { session: false }), checkGameInPlay(), async (req, res) => {
    const timestamp = req.body.timestamp;
    const songID = req.body.songID;
    const gameID = req.body.gameID;

    console.info(`Processing webcam image. songID=${songID}, timestamp=${timestamp}, gameID=${gameID}`);

    const base64Data = req.body.image.replace(/^data:image\/webp;base64,/, "");
    const truthData = await getPoseData(songID, timestamp)
  
    const userData = await processImageToPose(base64Data, timestamp)
    const user_joint_map = userData.points
    const truth_joint_map = truthData[0].data.joint_map

    console.debug("truth_joint_map", truth_joint_map);
    console.debug("user_joint_map", user_joint_map);
    
    results = pose.compare_poses(user_joint_map, truth_joint_map)

    user_joints_scaled = math.dotMultiply(user_joint_map, SCALE)
    truth_joints_scaled = math.dotMultiply(truth_joint_map, SCALE + 300)

    console.debug("truth_joints_scaled", truth_joints_scaled);
    console.debug("user_joints_scaled", user_joints_scaled);

    totalScore = results.reduce((a, b) => a + b, 0)
    console.info(`gameID=${gameID}, results=${results}, totalScore=${totalScore}`);

    res.send({
        "joints": user_joints_scaled,
        "mapping": userData.mapping,
        "truth_joints": truth_joints_scaled,
        "results": results,
        "score": totalScore
    });

    increaseScore(gameID, results, totalScore);

    // Debugging code to see image
    // require("fs").writeFile("out.png", base64Data, 'base64');
});

// Increase scores to a game record (IN GAME)
const increaseScore = (gameID, scores, scoreInc) => {
    let count = {"500":0, "300":0, "100":0, "0":0, "-1":0};

    for (let i = 1; i < scores.length; i++) {
        let num = scores[i];
        count[num.toString()] += 1;
    }
    
    gameCollection.updateOne({
        gameID: gameID
    }, {
        $inc: { 
            score: scoreInc,
            'scoreBreakdown.score500': count["500"],
            'scoreBreakdown.score300': count["300"],
            'scoreBreakdown.score100': count["100"],
            'scoreBreakdown.scoreX': count["0"]
        },
    }).catch(err => console.error("increaseScore - DB Query issue", err));
}

// Create Game request returns gameID
app.post('/createGame', passport.authenticate('bearer', { session: false }), (req, res) => {
    const songID = req.body.songID;
    const lobbyID = req.body.lobbyID;
    const playerID = req.user.id;
    const gameID = uuidv4();

    console.info(`Creating game for playerID=${playerID}, gameID=${gameID}`);

    // Note MONGODB has UUID support
    gameCollection.insertOne({
        songID: songID,
        gameID: gameID,
        lobbyID: lobbyID,
        playerID: mongoose.Types.ObjectId(playerID),
        score: 0,
        scoreBreakdown: {
            score500: 0,
            score300: 0,
            score100: 0,
            scoreX: 0
        },
        accuracy: 0.00,
        active: true,
        created: new Date()
    }).then((acknowledged, insertedId) => {
        if (acknowledged) {
            res.status(201).send({
                "message": "Game has been created",
                "gameID": gameID
            });
        }
    }).catch(err => console.error("/createGame - DB Query issue",  req.body.username, err));
});

// Multiplayer - get scores of all players in the lobby
app.get('/getLobbyScores', passport.authenticate('bearer', { session: false }), (req, res) => {
    lobbyID = req.query.lobbyID;

    gameCollection.aggregate([
        {
            $lookup: {
                from: "users",
                localField: "playerID",
                foreignField: "_id",
                as: "user"
            }
        },{
            $unwind: "$user"
        },{
            $match: {
                "lobbyID": lobbyID,
                "active": true
            }
        },{
            $project: {
                _id: 0,
                gameID: 1,
                score: 1,
                "user.username": 1,
                "user.profilePic": 1
            }
        }]).toArray().then(results => {
            res.status(200).send(results);
        }).catch(err => console.error("/getLobbyScores - DB Query issue", err));
});

// Finish a ongoing game and "save scores"
app.post('/finishGame', passport.authenticate('bearer', { session: false }), (req, res) => {
    gameID = req.body.gameID;

    gameCollection.findOneAndUpdate({
        gameID: gameID
    }, {
        $set: { active: false }
    }, (err, documents) => {
        if(err){ res.sendStatus(500); }
        if(documents) { res.sendStatus(200) }
    });
});

// Get list of all songs from DB
app.get('/songs', passport.authenticate('bearer', { session: false }), (req, res) => {
    songCollection.find().toArray()
        .then(results => {
            res.status(200).send(results);
        }).catch(err => console.error("/songs - DB Query issue", err));
});

// Get song details based on song ID
app.get('/songDetails', passport.authenticate('bearer', { session: false }), (req, res) => {
    const songID = req.query.songID;

    songCollection.findOne({ _id: parseInt(songID) }).then(results => {
        res.status(200).send(results);
    }).catch(err => console.error("/songDetails - DB Query issue", err));
});

// Gets results for the leaderboard
app.get('/getResults', passport.authenticate('bearer', { session: false }), (req, res) => {
    const gameID = req.query.gameID;
    const lobbyID = req.query.lobbyID;

    // Equivalent to SQL JOINs on collections
    if(lobbyID === undefined || lobbyID === null) {
        gameCollection.aggregate([
            {
                $lookup: {
                    from: "users",
                    localField: "playerID",
                    foreignField: "_id",
                    as: "user"
                }
            },{
                $lookup: {
                    from: "songs",
                    localField: "songID",
                    foreignField: "_id",
                    as: "songData"
                }
            },{
                $match: {
                    "gameID": gameID
                }
            },{
                $unwind: "$user"
            },{
                $unwind: "$songData"
            },{
                $project: {
                    _id: 0,
                    gameID: 1,
                    score: 1,
                    scoreBreakdown: 1,
                    "songData.name": 1,
                    "user.username": 1,
                    "user.profilePic": 1
                }
            },{ 
                $sort: {
                    score: -1
                }
            },{ 
                $limit: 20
            }]).toArray().then(result => {
                res.send(result);
            }).catch(err => console.error("/getResults - DB Query issue", err));
    }else {
        console.log("ELSE")
        gameCollection.aggregate([
            {
                $lookup: {
                    from: "users",
                    localField: "playerID",
                    foreignField: "_id",
                    as: "user"
                }
            },{
                $lookup: {
                    from: "songs",
                    localField: "songID",
                    foreignField: "_id",
                    as: "songData"
                }
            },{
                $match: {
                    "lobbyID":  lobbyID
                }
            },{
                $unwind: "$user"
            },{
                $project: {
                    _id: 0,
                    gameID: 1,
                    score: 1,
                    scoreBreakdown: 1,
                    "songData.name": 1,
                    "user.username": 1,
                    "user.profilePic": 1
                }
            },{ 
                $sort: {
                    score: -1
                }
            },{ 
                $limit: 20
            }]).toArray().then(result => {
                res.send(result);
            }).catch(err => console.error("/getResults - DB Query issue", err));
    }
   
});

// Multiplayer - Create a lobby, returns lobbyID
app.post('/createLobby', passport.authenticate('bearer', { session: false }), (req, res) => {
    const lobbyID = uuidv4();
    // Note MONGODB has UUID support
    lobbyCollection.insertOne({
        lobbyID: lobbyID
    }).then((acknowledged, insertedId) => {
        // TODO: add expiry
        if (acknowledged) {
            res.status(201).send({
                "message": "Lobby has been created",
                "lobbyID": lobbyID
            });
        }
    }).catch(err => console.error("/createLobby - DB Query issue", err));
})

//AUDIO AND VIDEO

// Gets video stream of dance/song based on ID
app.get('/songs/:id', (req, res) => {
    const id = req.params.id;
    // ms.pipe(req, res, "./dance_cover.mp4");
    ms.pipe(req, res, "../ml_backend/video.mp4");
});

// Gets audio stream of song based on ID for displaying
app.get('/audio', (req, res) => {
    const id = req.query.songID;
    // ms.pipe(req, res, `../frontend/src/components/${id}.mp3`);
    ms.pipe(req, res, "../frontend/src/components/audio2.mp3");
})


//////////////////////////////
//  Chatroom Implementation //
//////////////////////////////
const {
    userJoin,
    getCurrentUser,
    userLeave,
    getRoomUsers
} = require('./socketUtils');

socketIo.on('connection', (socket) => {
    // User joins room based on lobbyID
    socket.on('joinRoom', async (data) => {
        const token = data.token;
        var username = "";
        jwt.verify(token, process.env.private_key, function (err, decoded) {
            if (err) { console.error(err); return; }
            if (decoded) {
                username = decoded.username;
            }
        });

        const user = await userJoin(socket.id, username, data.lobbyID);
        socket.join(user.lobbyID);

        // Welcome current user
        socket.emit('message', 'Welcome to the lobby');

        // Broadcast when a user connects
        socket.broadcast.to(user.lobbyID).emit('message', `${user.username} has joined the lobby`);
        
        const roomUsers = await getRoomUsers(user.lobbyID);
        // Send users and room info
        socketIo.to(user.lobbyID).emit('roomUsers', {
            users: roomUsers
        });
    });

    // Listen for chatMessage
    socket.on('chatMessage', async (msg) => {
        const user = await getCurrentUser(socket.id);
        socketIo.to(user.lobbyID).emit('message', user.username + ": " + msg);
    });

    // Listen for event message - Start or change song
    socket.on('event', async (msg) => {
        const user = await getCurrentUser(socket.id);
        if (msg.start) {
            socket.to(user.lobbyID).emit('event', "START");
        } else if (msg.songID) {
            socketIo.to(user.lobbyID).emit('event', parseInt(msg.songID));
        }

    });

    // User disconnects (Leave lobby)
    socket.on('disconnect', async () => {
        if (!socket.id) {
            return;
        }

        const user = await userLeave(socket.id);

        if (user) {
            socketIo.to(user.lobbyID).emit(
                'message', `${user.username} has left the chat`
            );
            
            const roomUsers = await getRoomUsers(user.lobbyID);
            // Send users and room info
            socketIo.to(user.lobbyID).emit('roomUsers', {
                users: roomUsers,
            });
        }
    });
});

// Log to show server has started
// process.argv[2] is when using with pm2 for NGINGX
if (process.env.NODE_ENV != "test") {
    server.listen(process.argv[2] || port, () => {
        console.log(`Server listening on port:${process.argv[2] || port}`);
    });
}

module.exports = app;


