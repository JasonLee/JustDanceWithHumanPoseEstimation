import React, { useState } from 'react';
import axios from 'axios';
import styles from './css/Register.module.css';

export default function Register() {
    const [validUser, setValidUser] = useState({value: false});
    const [username, setUserName] = useState();
    const [password, setPassword] = useState();
    const [comfPassword, setComfPassword] = useState();
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
        }).catch(error => {
            if (error.response.status == 409) {
                setValidUser(false);
            }
        });
    }

    return (
        <div className={styles.RegisterWrapper}>
            <h1 className={styles.Container}>Register for an Account</h1>
            <form onSubmit={handleSubmit}>
                <div className={styles.Container}>
                    <label>
                        <label><b>Username</b></label>
                        <input className={styles.input} type="text" onChange={e => setUserName(e.target.value)} required="required"/>
                        {validUser ? null : <text>Username already exists</text>}
                    </label>
                    <label>
                        <label><b>Password</b></label>
                        <input className={styles.input} type="password" onChange={e => setPassword(e.target.value)} required="required"/>
                    </label>
                    <label>
                        <label><b>Confirm Password</b></label>
                        <input className={styles.input} type="password" onChange={e => setComfPassword(e.target.value)} required="required"/>
                        {password && comfPassword && password != comfPassword ? <div>Password does not match</div> : null}
                    </label>
                </div>
                <div className={styles.Container}>
                    <button className={styles.SubmitButton}type="submit">Submit</button>
                </div>
            </form>
        </div>
    )
}