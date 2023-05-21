document.querySelector("#myForm").addEventListener("submit", function (e) {
  e.preventDefault(); // Prevent form submission

  // Get form data
  var formData = new FormData(this);

  // Extract form data into separate variables
  var loginTimestamp = formData.get("loginTimestamp");
  var userID = formData.get("userID");
  var ipAddress = formData.get("ipAddress");
  var country = formData.get("country");
  var region = formData.get("region");
  var city = formData.get("city");
  var browserName = formData.get("browserName");
  var deviceType = formData.get("deviceType");
  var loginSuccessful = formData.get("loginSuccessful") === "on";

  // Construct the data object
  var data = {
    loginTimestamp: loginTimestamp,
    userID: userID,
    ipAddress: ipAddress,
    country: country,
    region: region,
    city: city,
    browserName: browserName,
    deviceType: deviceType,
    loginSuccessful: loginSuccessful,
  };

  // Send the form data to the Django backend
  fetch("http://127.0.0.1:8000/anomaly/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then(function (responseData) {
      // Display the result in the result area
      var resultText = "";
      // Modify this section based on the response data structure from your Django backend
      for (var key in responseData) {
        resultText += key + ": " + responseData[key] + "<br>";
      }
      document.querySelector(".result-area").innerHTML = resultText;
    })
    .catch(function (error) {
      console.log("Error:", error);
    });
});
