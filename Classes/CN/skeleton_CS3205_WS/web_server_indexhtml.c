/*
CS3205 assignment
This example serves an index.html file that references embedded image.jpg and style.css files. The server embeds the content of these files directly in the HTML response. Adjust the file paths and content types.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <time.h>

#define PORT 8080
#define BUFFER_SIZE 1024

void send_response(int client_socket, const char *response)
{
    send(client_socket, response, strlen(response), 0);
}

void handle_request(int client_socket, const char *request)
{
    char method[10];
    char path[256];

    // Parse HTTP request
    sscanf(request, "%s %s", method, path);

    if (strcmp(method, "GET") == 0)
    {
        if (strcmp(path, "/") == 0 || strcmp(path, "/post.html") == 0)
        {
            // Serve index.html with embedded images and CSS
            FILE *file = fopen("post.html", "r");
            if (file != NULL)
            {
                char buffer[BUFFER_SIZE];
                size_t bytesRead;

                // Send HTTP header
                const char *header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";
                send_response(client_socket, header);

                // Send file content
                while ((bytesRead = fread(buffer, 1, sizeof(buffer), file)) > 0)
                {
                    send(client_socket, buffer, bytesRead, 0);
                }

                fclose(file);
            }
            else
            {
                // Error opening file
                const char *error_response = "HTTP/1.1 500 Internal Server Error\r\n\r\nInternal Server Error";
                send_response(client_socket, error_response);
            }
        }
        else if (strcmp(path, "/image.jpg") == 0)
        {
            // Serve embedded image.jpg
            FILE *file = fopen("image.jpg", "rb");
            if (file != NULL)
            {
                char buffer[BUFFER_SIZE];
                size_t bytesRead;

                // Send HTTP header
                const char *header = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n\r\n";
                send_response(client_socket, header);

                // Send file content
                while ((bytesRead = fread(buffer, 1, sizeof(buffer), file)) > 0)
                {
                    send(client_socket, buffer, bytesRead, 0);
                }

                fclose(file);
            }
            else
            {
                // Error opening file
                const char *error_response = "HTTP/1.1 500 Internal Server Error\r\n\r\nInternal Server Error";
                send_response(client_socket, error_response);
            }
        }
        else if (strcmp(path, "/style.css") == 0)
        {
            // Serve embedded style.css
            FILE *file = fopen("style.css", "r");
            if (file != NULL)
            {
                char buffer[BUFFER_SIZE];
                size_t bytesRead;

                // Send HTTP header
                const char *header = "HTTP/1.1 200 OK\r\nContent-Type: text/css\r\n\r\n";
                send_response(client_socket, header);

                // Send file content
                while ((bytesRead = fread(buffer, 1, sizeof(buffer), file)) > 0)
                {
                    send(client_socket, buffer, bytesRead, 0);
                }

                fclose(file);
            }
            else
            {
                // Error opening file
                const char *error_response = "HTTP/1.1 500 Internal Server Error\r\n\r\nInternal Server Error";
                send_response(client_socket, error_response);
            }
        }
        else
        {
            // File not found
            const char *error_response = "HTTP/1.1 404 Not Found\r\n\r\nNot Found";
            send_response(client_socket, error_response);
        }
    }
    else if (strcmp(method, "POST") == 0)
    {

        time_t rawtime;
        struct tm *timeinfo;

        time(&rawtime);
        timeinfo = localtime(&rawtime);
        const char *response = "HTTP/1.1 200 OK\r\n\r content-type: POST \r\n\r";
        strcat(response,"Hi there! ");
        strcat(response,)
        
        printf("Current local time and date: %s\n", asctime(timeinfo));
        
    }
    else
    {
        // Method not supported
        const char *error_response = "HTTP/1.1 501 Not Implemented\r\n\r\nMethod not implemented";
        send_response(client_socket, error_response);
    }
}

int main()
{
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1)
    {
        perror("Error creating socket");
        exit(EXIT_FAILURE);
    }

    // Initialize server address structure
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(1);
    }

    // Bind the socket
    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
    {
        perror("Error binding socket");
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_socket, 5) == -1)
    {
        perror("Error listening for connections");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d...\n", PORT);

    while (1)
    {
        // Accept a connection
        client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &addr_len);
        if (client_socket == -1)
        {
            perror("Error accepting connection");
            continue;
        }

        printf("Connection accepted from %s:%d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

        // Read the HTTP request
        char buffer[BUFFER_SIZE];
        ssize_t bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
        if (bytes_received > 0)
        {
            buffer[bytes_received] = '\0';
            printf("Received request:\n%s\n", buffer);

            // Handle the request
            handle_request(client_socket, buffer);
        }

        // Close the client socket
        close(client_socket);
    }

    // Close the server socket (this part is unreachable in this simple example)
    close(server_socket);

    return 0;
}
