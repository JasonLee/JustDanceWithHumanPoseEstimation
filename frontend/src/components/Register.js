import React, { useState } from 'react';
import axios from 'axios';

export default function Register() {
    const [validUser, setValidUser] = useState({value: false});
    const [username, setUserName] = useState();
    const [password, setPassword] = useState({value: ""});
    const [comfPassword, setComfPassword] = useState({value: ""});
    const [email, setEmail] = useState();

    const handleSubmit = async e => {
        e.preventDefault();

        if (password && comfPassword && password != comfPassword) {
            return;
        }

        axios.post("/register", {
            username: username,
            password: password,
            email: email
        }).then(res => {
            return res;
        }).catch((error) => {
            if (error.response.status == 409) {
                setValidUser(false);
            }
        });
    }

    return (
        <div className="login-wrapper">
            <h1>Register for an Account</h1>
            <form onSubmit={handleSubmit}>
                <label>
                    <p>Username</p>
                    <input type="text" onChange={e => setUserName(e.target.value)} />
                    {validUser ? null : <text>Username already exists</text>}
                </label>
                <label>
                    <p>Password</p>
                    <input type="password" onChange={e => setPassword(e.target.value)} />
                </label>
                <label>
                    <p>Confirm Password</p>
                    <input type="password" onChange={e => setComfPassword(e.target.value)} />
                    {password && comfPassword && password != comfPassword 
                        ? <div>Password does not match</div> : null}
                </label>
                <div>
                    <button type="submit">Submit</button>
                </div>
            </form>
        </div>
    )
}