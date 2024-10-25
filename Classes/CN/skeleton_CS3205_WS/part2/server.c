#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define PORT 8080
#define BUFFER_SIZE 1024

void send_response(int client_socket, const char *status, const char *content_type, const char *body)
{
    char response[BUFFER_SIZE];
    snprintf(response, sizeof(response),
             "HTTP/1.1 %s\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %lu\r\n"
             "Connection: close\r\n"
             "\r\n"
             "%s",
             status, content_type, strlen(body), body);
    send(client_socket, response, strlen(response), 0);
}

void handle_get_request(int client_socket, const char *file_path)
{
    FILE *file = fopen(file_path, "r");
    if (file)
    {
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        char *file_content = malloc(file_size + 1);
        fread(file_content, 1, file_size, file);
        file_content[file_size] = '\0';
        fclose(file);
        send_response(client_socket, "200 OK", "text/html", file_content);
        free(file_content);
    }
    else
    {
        send_response(client_socket, "404 Not Found", "text/html", "<h1>404 Not Found</h1>");
    }
}

void handle_post_request(int client_socket, const char *data)
{
    printf("POST data: %s\n", data);
    send_response(client_socket, "200 OK", "text/html", "<h1>POST Received</h1>");
}

int main()
{
    int server_fd, client_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE] = {0};

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    while (1)
    {
        if ((client_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0)
        {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        read(client_socket, buffer, BUFFER_SIZE);

        char *method = strtok(buffer, " ");
        char *path = strtok(NULL, " ");
        strtok(NULL, "\n"); // Skipping HTTP version

        if (strcmp(method, "GET") == 0)
        {
            if (strcmp(path, "/") == 0)
            {
                handle_get_request(client_socket, "index.html");
            }
            else if (strcmp(path, "/login") == 0)
            {
                handle_get_request(client_socket, "login.html");
            }
            else
            {
                handle_get_request(client_socket, "404.html");
            }
        }
        else if (strcmp(method, "POST") == 0)
        {
            if (strcmp(path, "/login") == 0)
            {
                strtok(NULL, "\n"); // Skipping headers
                char *post_data = strtok(NULL, "\r");
                handle_post_request(client_socket, post_data);
            }
            else
            {
                handle_get_request(client_socket, "404.html");
            }
        }

        close(client_socket);
    }

    return 0;
}
