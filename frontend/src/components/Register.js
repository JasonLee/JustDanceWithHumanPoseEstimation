import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './css/Register.module.css';
import { Link } from 'react-router-dom';

// Register Page
export default function Register() {
    const [validUser, setValidUser] = useState({value: false});
    const [username, setUserName] = useState();
    const [password, setPassword] = useState();
    const [redirect, setRedirect] = useState();
    const [comfPassword, setComfPassword] = useState();
    const [image, setImage] = useState();
    const [email, setEmail] = useState("");

    const [passwordValid, setPasswordValid] = useState(false);
    const [emailValid, setEmailValid] = useState(false);
    
    // Convert uploaded image to base64; ready to send
    const handleImageUpload = e => {
      const [file] = e.target.files;
      
      if (file) {
        const reader = new FileReader();

        reader.onload = (e) => {
            setImage(e.target.result);
        }
        reader.readAsDataURL(file);

      }
    };

    // Password validation onChange
    const validatePassword = (password) => {
        // REGEX for 1 digit, 1 lowercase, 1 uppercase, 8 characters
        const passwordCheck = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[0-9a-zA-Z]{8,}$/;
  
        setPasswordValid(passwordCheck.test(password));
        setPassword(password);
    }

    // Email validation onChange
    const validateEmail = (email) => {
        const emailCheck = /^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])/;
        setEmailValid(emailCheck.test(email));
        setEmail(email);
    }

    // Submit button
    const handleSubmit = async e => {
        e.preventDefault();

        if (password && comfPassword && password != comfPassword) {
            return;
        }

        axios.post("/register", {
            username: username,
            password: password,
            email: email,
            profilePic: image
        }).then(res => {
            setRedirect(true);
            return;
        }).catch(error => {
            if (error.response.status == 409 && error.response.data == "Error: Unable to create account") {
                setValidUser(false);
            }
        });
    };

    // When registration submitted, swap to message
    if (redirect) {
        return (
            <div style={{color: "white", fontSize:"20px"}}>
                Please verify your account. <br />
                An verification email has been sent.
                <br />
                <br />
                <Link to='/'>
                    Go to Home
                </Link>
            </div>
        
        );
    }

    return (
        <div className={styles.RegisterWrapper}>
            <h1 className={styles.Container}>Register for an Account</h1>
            <form onSubmit={handleSubmit}>
                <div className={styles.Container}>

                    <label className={styles.box}>
                        <label><b>Username</b></label>
                        <input className={styles.input} type="text" onChange={e => setUserName(e.target.value)} required="required"/>
                        {validUser ? null : <div><div style={{color: "red"}}>Username already exists </div>  <br /></div> }
                    </label>
                    <label>
                        <label><b>Password</b></label>
                        <input className={styles.input} type="password" onChange={e =>  validatePassword(e.target.value)} required="required"/>
                        {passwordValid ? null : <div><div style={{color: "red"}}>Password must contain atleast 1 digit, 1 lowercase, 1 uppercase and be 8 characters long </div>  <br /></div> }
                    </label>
                    <label>
                        <label><b>Confirm&nbsp;Password</b></label>
                        <input className={styles.input} type="password" onChange={e => setComfPassword(e.target.value)} required="required"/>
                        {password && comfPassword && password != comfPassword ? <div><div style={{color: "red"}}>Password does not match</div><br /> </div> : null}
                    </label>
                    <label>
                        <label><b>Email&nbsp;Address</b></label>
                        <input className={styles.input} type="email" onChange={e => validateEmail(e.target.value)} required="required"/>
                        {emailValid ? null : <div><div style={{color: "red"}}>Please enter a valid email </div>  <br /></div> }
                    </label>
                    <label>
                        <label><b>Profile&nbsp;Picture (Optional)</b></label> <br />
                        <input type="file" accept="image/*" onChange={handleImageUpload} />
                        {image && <div className={styles.picWrapper}>
                            <img src={image} className={styles.pic} />
                        </div>
                        }
                    </label>
                </div>
                <div className={styles.Container}>
                    <button className={styles.SubmitButton} type="submit" disabled={!emailValid || !passwordValid || password != comfPassword}>Submit</button>
                </div>
            </form>
        </div>
    )
}