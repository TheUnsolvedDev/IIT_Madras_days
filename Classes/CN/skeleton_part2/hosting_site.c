#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFSIZE 1024

void handle_get_request(int client_socket)
{
    const char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body><h1>Hello, World!</h1></body></html>";
    send(client_socket, response, strlen(response), 0);
}

void handle_request(int client_socket)
{
    char buffer[BUFSIZE];
    recv(client_socket, buffer, BUFSIZE, 0);

    // Check if it's a GET request
    if (strncmp(buffer, "GET", 3) == 0)
    {
        handle_get_request(client_socket);
    }
    else
    {
        // Handle other request types if needed
        const char *response = "HTTP/1.1 501 Not Implemented\r\nContent-Type: text/plain\r\n\r\nNot Implemented";
        send(client_socket, response, strlen(response), 0);
    }

    close(client_socket);
}

int main()
{
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_size = sizeof(struct sockaddr_in);

    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);

    // Set up server address structure
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket
    bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr));

    // Listen for incoming connections
    listen(server_socket, 10);

    printf("Server listening on port %d...\n", PORT);

    while (1)
    {
        // Accept connection
        client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &addr_size);

        if (fork() == 0)
        {                         // Fork to handle multiple connections
            close(server_socket); // Close in child process

            // Handle the request
            handle_request(client_socket);

            exit(0); // Terminate child process
        }
        else
        {
            close(client_socket); // Close in parent process
        }
    }

    return 0;
}
