#include <stdio.h>
#include <chrono>
#include <csignal>
#include <fstream>

#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <jetgpio.h>

#define BEAM_START_EVENT_ID -10
#define BEAM_STOP_EVENT_ID -11
#define ARDUINO_EVENT_ID -12

/* Global variables */
unsigned long beam_start_timestamp;
unsigned long beam_stop_timestamp;
unsigned long arduino_timestamp;
time_t time_convert;
uint64_t app_start_timestamp = 0, app_end_timestamp = 0;

/* Variable for trigger GPIO pin */
int beam_start_gpio = 7;
int beam_stop_gpio = 11;
int arduino_gpio = 12;

/* --------------------------- */
/* Beam Line Trigger Functions */
/* --------------------------- */

/* Function to handle RISING edge beam START trigger */
void beam_line_start_trigger() {
    time_t time = static_cast<time_t>(beam_start_timestamp/1e9); // Convert nanoseconds to seconds
    fprintf(stdout, "Beam START signal timestamp: %lu [us], %s\n", (unsigned long)(beam_start_timestamp/1e3), ctime(&time));
    fflush(stdout); // Flush stdout to ensure immediate output
}

/* Function to handle RISING edge beam STOP trigger */
void beam_line_stop_trigger() {
    time_t time = static_cast<time_t>(beam_stop_timestamp/1e9); // Convert nanoseconds to seconds
    fprintf(stdout, "Beam STOP signal timestamp: %lu [us], %s\n", (unsigned long)(beam_stop_timestamp/1e3), ctime(&time));
    fflush(stdout); // Flush stdout to ensure immediate output
}

/* Function to handle RISING edge Arduino trigger */
void arduino_trigger() {
    time_t time = static_cast<time_t>(arduino_timestamp/1e9); // Convert nanoseconds to seconds
    fprintf(stdout, "Arduino signal timestamp: %lu [us], %s\n", (unsigned long)(arduino_timestamp/1e3), ctime(&time));
    fflush(stdout); // Flush stdout to ensure immediate output
}

/* ---------------------- */
/* Exit Handler Functions */
/* ---------------------- */

// Signal handler for Ctrl+C
void signal_handler(int signum) {
    fprintf(stdout, "\nUser pressed Ctrl+C. Exiting program...\n");
    fflush(stdout); // Flush stdout to ensure immediate output
    exit(0);
}

void atexit_handler() {
    fprintf(stdout, "\n# ---------------------------------------------- #\n");
    fprintf(stdout, "# --- Exit application and cleanup resources --- #\n");
    fprintf(stdout, "# ---------------------------------------------- #\n\n");

    // Release peripherals
    fprintf(stdout,  "Release GPIO.\n");
    gpioTerminate();

    uint64_t exit_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(exit_time/1e6); // Convert microseconds to seconds
    fprintf(stdout, "\nProgram exit timestamp: %lu [us], %s", exit_time, ctime(&time_convert));
    fprintf(stdout,  "\n--------------------------------------------------------------------------\n");
    fflush(stdout); // Flush stdout to ensure immediate output
}

/* ------------- */
/* Main Function */
/* ------------- */

int main(int argc, char **argv){
    fprintf(stdout,  "\n--------------------------------------------------------------------------\n");
    // Print start timestamp
    app_start_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(app_start_timestamp/1e6); // Convert microseconds to seconds
    fprintf(stdout, "Program start timestamp: %lu [us], %s", app_start_timestamp, ctime(&time_convert));

    /* ------------------------------ */
    /* Setup signal and exit handlers */
    /* ------------------------------ */
    fprintf(stdout, "\n# ------------------------------------------- #\n");
    fprintf(stdout, "# --- Setting up signal and exit handlers --- #\n");
    fprintf(stdout, "# ------------------------------------------- #\n\n");

    // Register atexit handler to ensure it gets called on normal exit
    const int result = std::atexit(atexit_handler); // Handler will be called

    // Register signal handler for Ctrl+C (SIGINT)
    std::signal(SIGINT, signal_handler);

    if (result != 0)
    {
        fprintf(stdout, "atexit registration failed\n");
        return EXIT_FAILURE;
    }

    if (getuid() != 0)
    {
        fprintf(stdout, "Root privileges are required for beam signal reception. Run the program with sudo. Exiting...\n");
        exit(-1);
    }

    /* ----------------------------------------------- */
    /* Setup trigger signal from beam line and Arduino */
    /* ----------------------------------------------- */
    fprintf(stdout, "\n# ---------------------------------------------- #\n");
    fprintf(stdout, "# --- Setup beam and Arduino trigger signals --- #\n");
    fprintf(stdout, "# ---------------------------------------------- #\n\n");

    // Prepare variables for GPIO setup
    int Init;
    int stat;

    // Intitalize Orin GPIO library
    fprintf(stdout,  "GPIO library initialization...\n");
    Init = gpioInitialise();
    if (Init < 0) {
        fprintf(stdout, "Jetgpio initialisation failed. Error code:  %d\n", Init);
        exit(1);
    }

    // Set up GPIO pin for beam START trigger
    fprintf(stdout,  "GPIO pin %d setup for beam START trigger...\n", beam_start_gpio);
    stat = gpioSetMode(beam_start_gpio, JET_INPUT); // Set GPIO pin as input
    if (stat < 0)
    {
        fprintf(stdout, "Failed to set pin mode for beam START trigger GPIO %d. Error code: %d\n", beam_start_gpio, stat);
        exit(1);
    }
    // Set up interrupt handler for EITHER edge on the beam START trigger GPIO pin
    fprintf(stdout,  "Interrupt handler setup for EITHER edge beam START trigger...\n");
    stat = gpioSetISRFunc(beam_start_gpio, EITHER_EDGE, 10 /* us */, &beam_start_timestamp, &beam_line_start_trigger);
    if (stat < 0)
    {
        fprintf(stdout, "Failed to set alert function for EITHER edge beam START trigger GPIO %d. Error code: %d\n", beam_start_gpio, stat);
        exit(1);
    }
    fprintf(stdout, "GPIO pin %d set up for EITHER edge beam START trigger.\n", beam_start_gpio);

    // Set up GPIO pin for beam STOP trigger
    fprintf(stdout,  "GPIO pin %d setup for STOP trigger...\n", beam_stop_gpio);
    stat = gpioSetMode(beam_stop_gpio, JET_INPUT); // Set GPIO pin as input
    if (stat < 0)
    {
        fprintf(stdout, "Failed to set pin mode for beam STOP trigger GPIO %d. Error code: %d\n", beam_stop_gpio, stat);
        exit(1);
    }
    // Set up interrupt handler for EITHER edge on the beam STOP trigger GPIO pin
    fprintf(stdout,  "Interrupt handler setup for EITHER edge beam STOP trigger...\n");
    stat = gpioSetISRFunc(beam_stop_gpio, EITHER_EDGE, 10 /* us */, &beam_stop_timestamp, &beam_line_stop_trigger);
    if (stat < 0)
    {
        fprintf(stdout, "Failed to set alert function for EITHER edge beam STOP trigger GPIO %d. Error code: %d\n", beam_stop_gpio, stat);
        exit(1);
    }
    fprintf(stdout, "GPIO pin %d set up for EITHER edge beam STOP trigger.\n", beam_stop_gpio);

    // Set up GPIO pin for Arduino signal from beam line
    fprintf(stdout,  "GPIO pin %d setup for Arduino signal...\n", arduino_gpio);
    stat = gpioSetMode(arduino_gpio, JET_INPUT); // Set GPIO pin as input
    if (stat < 0)
    {
        fprintf(stdout, "Failed to set pin mode for Arduino GPIO %d. Error code: %d\n", arduino_gpio, stat);
        exit(1);
    }
    // Set up interrupt handler for RISING edge Arduino trigger GPIO pin
    fprintf(stdout,  "Interrupt handler setup for Arduino signal...\n");
    stat = gpioSetISRFunc(arduino_gpio, RISING_EDGE, 10 /* us */, &arduino_timestamp, &arduino_trigger);
    if (stat < 0)
    {
        fprintf(stdout, "Failed to set alert function for Arduino RISING trigger GPIO %d. Error code: %d\n", arduino_gpio, stat);
        exit(1);
    }
    fprintf(stdout, "GPIO pin %d set up for Arduino RISING trigger signal from beam line.\n", arduino_gpio);

    // Infinite loop to keep the program running and waiting for beam line triggers
    fprintf(stdout, "\nWaiting for beam line triggers...\n");
    unsigned long timestamp = 0;
    while(true)
    {
        // Sleep for a short duration to reduce CPU usage
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        fprintf(stdout, "Timestamp: %lu [us]\r\n", timestamp);
        fflush(stdout); // Flush stdout to ensure immediate output
        sleep(30);
    }

    exit(0);
}
