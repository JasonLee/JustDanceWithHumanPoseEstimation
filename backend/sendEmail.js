const nodemailer = require('nodemailer');
require('dotenv').config();

// The credentials for the email account you want to send mail from. 
const credentials = {
    service: 'gmail',
    host: 'smtp.gmail.com',
    port: 465,
    secure: true,
    auth: {
        // These environment variables will be pulled from the .env file
        user: process.env.MAIL_USER,
        pass: 'HZ5$:GpnVD.hH<"F'
    }
}

// Getting Nodemailer all setup with the credentials for when the 'sendEmail()'
// function is called.
const transporter = nodemailer.createTransport(credentials);

module.exports = async (to, content) => {

    // The from and to addresses for the email that is about to be sent.
    const contacts = {
        from: process.env.MAIL_USER,
        subject: "Just Dance with Human Pose Estimation Registration",
        to
    }

    const email = Object.assign({}, content, contacts)

    await transporter.sendMail(email, function (error, info) {
        if (error) {
            console.log("Email failed to send", error);
        } else {
            console.log('Email sent successfully', info.response);
        }
    });
}