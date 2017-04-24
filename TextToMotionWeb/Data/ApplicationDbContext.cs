using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using TextToMotionWeb.Models;
using TextToMotionWeb.Data;
using Microsoft.Extensions.Configuration;
using Microsoft.AspNetCore.Builder;
using TextToMotionWeb.Services;

namespace TextToMotionWeb.Data
{
    public class ApplicationDbContext : IdentityDbContext<ApplicationUser, Role, string>
    {
        public DbSet<ApplicationUser> ApplicationUsers {get; set;}
        public DbSet<Role> Roles {get; set;}
        public DbSet<Tag> Tags { get; set; }
        public DbSet<Media> Media { get; set; }
        public DbSet<Image> Images { get; set; }
        public DbSet<Video> Videos { get; set; }
        public DbSet<ImageTag> ImageTags { get; set; }
        public DbSet<VideoTag> VideoTags { get; set; }
        public DbSet<UserMediaFeedback> UserMediaFeedback { get; set; }


        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
          var config = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json")
            .Build();

          optionsBuilder.UseMySql(@config.GetConnectionString("DefaultConnection"));
        }


        protected override void OnModelCreating(ModelBuilder builder)
        {
            base.OnModelCreating(builder);
            // Customize the ASP.NET Identity model and override the defaults if needed.
            // For example, you can rename the ASP.NET Identity table names and more.
            // Add your customizations after calling base.OnModelCreating(builder);

            builder.Entity<Media>()
                .Property(b => b.Inserted)
                .ValueGeneratedOnAdd();

            builder.Entity<Media>()
                .Property(b => b.Updated)
                .ValueGeneratedOnAddOrUpdate();

            builder.Entity<UserMediaFeedback>()
                .Property(b => b.Inserted)
                .ValueGeneratedOnAdd();

            builder.Entity<Tag>()
                .Property(b => b.Inserted)
                .ValueGeneratedOnAdd();



        }
    }
}
