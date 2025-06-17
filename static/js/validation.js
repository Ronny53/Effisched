const form = document.getElementById('form')
const username_input = document.getElementById('username-input')
const email_input = document.getElementById('email-input')
const password_input = document.getElementById('password-input')
const repeat_password_input = document.getElementById('repeat-password-input')
const error_message = document.getElementById('error-message')

form.addEventListener('submit', (e) => {
    e.preventDefault()  // Prevent form from submitting immediately

    let errors = []

    if(username_input){
        // If we have a username input, then we're in the signup
        errors = getSignupFormErrors(username_input.value, email_input.value, password_input.value, repeat_password_input.value)
    }
    else{
        // If we don't have a username input, we're in the login form
        errors = getLoginFormErrors(email_input.value, password_input.value)
    }

    if(errors.length > 0){
        // If there are any errors, display them
        error_message.innerText = errors.join(". ")
    } else {
        // If no errors, check which button was clicked for redirection
        const loginButton = document.querySelector('button[type="submit"]:not([id="signup-button"])');
        const signupButton = document.querySelector('button[type="submit"]:not([id="login-button"])');

        // Redirect based on the button clicked
        if (signupButton && signupButton.innerText.toLowerCase() === "sign-up") {
            window.location.href = "/welcome";
        } else if (loginButton && loginButton.innerText.toLowerCase() === "login") {
            window.location.href = "/calendar";
        }
    }
})

function getSignupFormErrors(username , email , password, repeatPassword){
    let errors = []

    if(username === '' || username == null){
        errors.push('Username is required')
        username_input.parentElement.classList.add('incorrect')
    }
    if(email === '' || email == null){
        errors.push('Email is required')
        email_input.parentElement.classList.add('incorrect')
    }
    if(password === '' || password == null){
        errors.push('Password is required')
        password_input.parentElement.classList.add('incorrect')
    }
    if(password.length < 8){
        errors.push('Password must have at least 8 characters')
        password_input.parentElement.classList.add('incorrect')
    }
    if(password !== repeatPassword){
        errors.push('Password does not match repeated password')
        password_input.parentElement.classList.add('incorrect')
        repeat_password_input.parentElement.classList.add('incorrect')
    }

    return errors
}

function getLoginFormErrors(email, password){
    let errors = []

    if(email === '' || email == null){
        errors.push('Email is required')
        email_input.parentElement.classList.add('incorrect')
    }
    if(password === '' || password == null){
        errors.push('Password is required')
        password_input.parentElement.classList.add('incorrect')
    }

    return errors
}

const allInputs = [username_input, email_input, password_input, repeat_password_input].filter(input => input != null)

allInputs.forEach(input => {
    input.addEventListener('input', () => {
        if(input.parentElement.classList.contains('incorrect')){
            input.parentElement.classList.remove('incorrect')
            error_message.innerText = ''
        }
    })
})
