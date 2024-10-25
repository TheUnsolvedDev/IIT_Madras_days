#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 8081
#define BUFFER_SIZE 1024

int main()
{
    char buffer[BUFFER_SIZE];
    char resp[] = "HTTP/1.1 200 OK\r\n"
                  "Server: webserver-c\r\n"
                  "Content-type: text/html\r\n\r\n"
                  "<!DOCTYPE html>\r\n"
                  "<html lang='en'>\r\n"
                  "<head>\r\n"
                  "<meta charset='UTF-8'>\r\n"
                  "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\r\n"
                  "<title>Login Page</title>\r\n"
                  "<script>\r\n"
                  "function displayGreeting() {\r\n"
                  "var userName = document.getElementById('name').value;\r\n"
                  "var serverTime = new Date().toLocaleTimeString();\r\n"
                  "var greetingMessage = 'Good morning ' + userName + '. Server time is ' + serverTime;\r\n"
                  "document.getElementById('greeting').innerText = greetingMessage;\r\n"
                  "}\r\n"
                  "</script>\r\n"
                  "</head>\r\n"
                  "<body>\r\n"
                  "<form>\r\n"
                  "<label for='name'>Name:</label>\r\n"
                  "<input type='text' id='name' name='name' required>\r\n"
                  "<button type='button' onclick='displayGreeting()'>Login</button>\r\n"
                  "</form>\r\n"
                  "<div id='greeting'></div>\r\n"
                  "</body>\r\n"
                  "</html>\r\n\r\n";

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        perror("webserver (socket)");
        return 1;
    }
    printf("socket created successfully\n");

    struct sockaddr_in host_addr;
    int host_addrlen = sizeof(host_addr);

    host_addr.sin_family = AF_INET;
    host_addr.sin_port = htons(PORT);
    host_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sockfd, (struct sockaddr *)&host_addr, host_addrlen) != 0)
    {
        perror("webserver (bind)");
        return 1;
    }
    printf("socket successfully bound to address\n");

    if (listen(sockfd, SOMAXCONN) != 0)
    {
        perror("webserver (listen)");
        return 1;
    }
    printf("server listening for connections\n");

    for (;;)
    {
        int newsockfd = accept(sockfd, (struct sockaddr *)&host_addr,
                               (socklen_t *)&host_addrlen);
        if (newsockfd < 0)
        {
            perror("webserver (accept)");
            continue;
        }
        printf("connection accepted\n");

        int valread = read(newsockfd, buffer, BUFFER_SIZE);
        if (valread < 0)
        {
            perror("webserver (read)");
            continue;
        }

        int valwrite = write(newsockfd, resp, strlen(resp));
        if (valwrite < 0)
        {
            perror("webserver (write)");
            continue;
        }

        close(newsockfd);
    }

    return 0;
}