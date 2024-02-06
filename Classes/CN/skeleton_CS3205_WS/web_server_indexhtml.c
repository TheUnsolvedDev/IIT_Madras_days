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

void send_file(int client_socket, const char *file_path, const char *content_type)
{
    FILE *file = fopen(file_path, "rb");
    if (file != NULL)
    {
        char buffer[BUFFER_SIZE];
        size_t bytes_read;

        char header[BUFFER_SIZE];
        sprintf(header, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\n\r\n", content_type);
        send_response(client_socket, header);

        while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0)
        {
            send(client_socket, buffer, bytes_read, 0);
        }

        fclose(file);
    }
    else
    {
        const char *error_response = "HTTP/1.1 500 Internal Server Error\r\n\r\nInternal Server Error";
        send_response(client_socket, error_response);
    }
}

char *extract_after_data(const char *buffer)
{
    const char *data_pos = strstr(buffer, "username");
    if (data_pos != NULL)
    {
        data_pos += strlen("username':");
        return strdup(data_pos);
    }
    return NULL;
}

void handle_request(int client_socket, const char *request)
{
    char method[10];
    char path[256];

    sscanf(request, "%s %s", method, path);

    if (strcmp(method, "GET") == 0)
    {
        if (strcmp(path, "/") == 0 || strcmp(path, "/login.html") == 0)
        {
            send_file(client_socket, "login.html", "text/html");
        }
        else if (strcmp(path, "/image.jpg") == 0)
        {
            send_file(client_socket, "images/image.jpg", "image/jpg");
        }
        else
        {
            FILE *file = fopen("404.html", "r");
            if (file != NULL)
            {
                char buffer[BUFFER_SIZE];
                size_t bytesRead;
                const char *error_response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\n\r\n";

                send_response(client_socket, error_response);

                while ((bytesRead = fread(buffer, 1, sizeof(buffer), file)) > 0)
                {
                    send(client_socket, buffer, bytesRead, 0);
                }

                fclose(file);
            }
            else
            {
                const char *error_response = "HTTP/1.1 500 Internal Server Error\r\n\r\nInternal Server Error";
                send_response(client_socket, error_response);
            }
        }
    }
    else if (strcmp(method, "POST") == 0)
    {
        time_t rawtime;
        struct tm *timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);

        const char *response = "HTTP/1.1 200 OK\r\n\r content-type: POST \r\n\r";
        send_response(client_socket, response);

        printf("The data is %s\n", extract_after_data(request));
        printf("Current local time and date: %s\n", asctime(timeinfo));
    }
    else
    {
        const char *error_response = "HTTP/1.1 501 Not Implemented\r\n\r\nMethod not implemented";
        send_response(client_socket, error_response);
    }
}

int main()
{
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1)
    {
        perror("Error creating socket");
        exit(EXIT_FAILURE);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(1);
    }

    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
    {
        perror("Error binding socket");
        exit(EXIT_FAILURE);
    }

    if (listen(server_socket, 5) == -1)
    {
        perror("Error listening for connections");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d...\n", PORT);

    while (1)
    {

        client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &addr_len);
        if (client_socket == -1)
        {
            perror("Error accepting connection");
            continue;
        }

        printf("Connection accepted from %s:%d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

        char buffer[BUFFER_SIZE];
        ssize_t bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
        if (bytes_received > 0)
        {
            buffer[bytes_received] = '\0';
            printf("Received request:\n%s\n", buffer);

            handle_request(client_socket, buffer);
        }
        close(client_socket);
    }
    close(server_socket);

    return 0;
}
