#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
	int value;
	struct node *next;
} node;

void print_list(node *head)
{
	node *temp = head;
	while (temp != NULL)
	{
		printf("%d --> ", temp->value);
		temp = temp->next;
	}
	printf("NULL\n");
}

void insert_at_beginning(node **head, int value)
{
	node *new_node = (node *)malloc(sizeof(node));
	if (new_node == NULL)
	{
		return;
	}
	new_node->value = value;
	new_node->next = *head;
	*head = new_node;
}

void insert_at_end(node **head, int value)
{
	node *new_node = (node *)malloc(sizeof(node));
	if (new_node == NULL)
	{
		return;
	}
	new_node->value = value;
	new_node->next = NULL;

	if (*head == NULL)
	{
		*head = new_node;
	}
	else
	{
		node *temp = *head;
		while (temp->next != NULL)
		{
			temp = temp->next;
		}
		temp->next = new_node;
	}
}

void insert_at_index(node **head, int index, int value)
{
	node *new_node = (node *)malloc(sizeof(node));
	if (new_node == NULL)
	{
		return;
	}
	new_node->value = value;

	if (index == 0)
	{
		new_node->next = *head;
		*head = new_node;
	}
	else
	{
		node *temp = *head;
		for (int i = 0; i < index - 1 && temp != NULL; ++i)
		{
			temp = temp->next;
		}
		if (temp == NULL)
		{
			free(new_node);
			return;
		}
		new_node->next = temp->next;
		temp->next = new_node;
	}
}

void delete_at_beginning(node **head)
{
	if (*head != NULL)
	{
		node *temp = *head;
		*head = (*head)->next;
		free(temp);
	}
}

void delete_at_end(node **head)
{
	if (*head != NULL)
	{
		if ((*head)->next == NULL)
		{
			free(*head);
			*head = NULL;
		}
		else
		{
			node *temp = *head;
			while (temp->next->next != NULL)
			{
				temp = temp->next;
			}
			free(temp->next);
			temp->next = NULL;
		}
	}
}

void delete_at_index(node **head, int index)
{
	if (*head != NULL && index >= 0)
	{
		if (index == 0)
		{
			node *temp = *head;
			*head = (*head)->next;
			free(temp);
		}
		else
		{
			node *temp = *head;
			for (int i = 0; i < index - 1 && temp != NULL; ++i)
			{
				temp = temp->next;
			}
			if (temp == NULL || temp->next == NULL)
			{
				return;
			}
			node *to_delete = temp->next;
			temp->next = temp->next->next;
			free(to_delete);
		}
	}
}

void delete_all(node **head)
{
	while (*head != NULL)
	{
		node *temp = *head;
		*head = (*head)->next;
		free(temp);
	}
}

int main()
{
	node *head = (node *)malloc(sizeof(node));
	head->value = 0;
	node *head_next = (node *)malloc(sizeof(node));
	head_next->value = 10;
	head->next = head_next;

	for (int i = 0; i < 10; i++)
		insert_at_beginning(&head, i);

	for (int i = 0; i < 10; i++)
		insert_at_end(&head, i);

	for (int i = 0; i < 10; i++)
		insert_at_index(&head, i, 100);

	for (int i = 0; i < 10; i++)
		delete_at_beginning(&head);

<<<<<<< Updated upstream
	print_list(head);
	delete_all(&head);
	return 0;
=======
	print_list(head);
	return 0;
>>>>>>> Stashed changes
}